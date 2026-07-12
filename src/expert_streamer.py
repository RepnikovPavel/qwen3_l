"""
Expert-level offload for Qwen3-MoE across the available GPUs.

Load path: device_map="cpu" (FP8 validator bypass), then manual placement.
Routed experts (27 GB stacked gate_up_proj/down_proj) stay on CPU (pinned).

Two paging modes (DEMO_EXPERT_SLICE_MODE, default "1"):
  slice — LRU-cache individual expert weight slices (~4.5 MiB each, top-8/layer
          ≈ 36 MiB). Copies only active experts per forward; 16× less data than
          paging whole Qwen3MoeExperts modules (576 MiB).
  layer — legacy: page entire experts module GPU<-CPU per layer (slow).

2-GPU split (DEMO_EXPERT_SPLIT_GPU=1): contiguous-half placement with rotary
bridge via layer pre-hooks (with_kwargs=True). Per-GPU LRU caches.
"""
from __future__ import annotations

import os
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertOffloader:
    """Keeps routed-expert weights on CPU; pages slices or modules per forward."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list = []
        self._installed = False
        self._home_gpu: dict[int, int] = {}
        self._cache_gib = float(os.environ.get("DEMO_EXPERT_CACHE_GIB", "11"))
        # Layer-module LRU (legacy mode)
        self._cache_used_bytes: dict[int, int] = {}
        self._cache_order: dict[int, list[int]] = {}
        self._on_gpu: dict[int, set[int]] = {}
        self._module_size: dict[int, int] = {}
        self._module_by_id: dict[int, nn.Module] = {}
        self._module_home_gpu: dict[int, int] = {}
        # Per-expert-slice LRU (default mode)
        self._slice_mode = os.environ.get("DEMO_EXPERT_SLICE_MODE", "1") != "0"
        self._slice_cache: dict[int, dict[tuple, tuple]] = {}   # gpu -> key -> (g,d,sz)
        self._slice_order: dict[int, list[tuple]] = {}
        self._slice_used: dict[int, int] = {}
        self._split_gpus = False

    @classmethod
    def install(cls, model: nn.Module) -> "ExpertOffloader":
        od = cls(model)
        od._place_and_hook()
        return od

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for experts in self._iter_expert_modules():
            orig = getattr(experts, "_qwen_orig_forward", None)
            if orig is not None:
                experts.forward = orig
        self._installed = False

    def _iter_expert_modules(self):
        inner = getattr(self.model, "model", self.model)
        for layer in getattr(inner, "layers", []):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is not None and hasattr(experts, "gate_up_proj"):
                yield experts

    def _init_gpu_cache(self, gpu: int):
        if gpu not in self._cache_order:
            self._cache_order[gpu] = []
            self._on_gpu[gpu] = set()
            self._cache_used_bytes[gpu] = 0
            self._slice_cache[gpu] = {}
            self._slice_order[gpu] = []
            self._slice_used[gpu] = 0

    @staticmethod
    def _pin_cpu(param: nn.Parameter) -> None:
        param.data = param.data.to("cpu", non_blocking=False)
        try:
            param.data = torch.empty(
                param.data.shape, dtype=param.data.dtype,
                device="cpu", pin_memory=True,
            ).copy_(param.data)
        except RuntimeError:
            pass

    def _place_and_hook(self):
        if self._installed:
            return

        n_gpus = torch.cuda.device_count() or 1
        split = (n_gpus >= 2
                 and os.environ.get("DEMO_EXPERT_SPLIT_GPU", "1") != "0")
        self._split_gpus = split

        inner = getattr(self.model, "model", self.model)
        layers = list(getattr(inner, "layers", []))
        n_layers = len(layers)
        half = n_layers // 2

        embed_home = "cuda:0" if split else "cuda:0"
        exit_home = f"cuda:{n_gpus - 1}" if split else "cuda:0"

        if hasattr(inner, "embed_tokens") and inner.embed_tokens is not None:
            inner.embed_tokens.to(embed_home)
        if hasattr(inner, "rotary_emb") and inner.rotary_emb is not None:
            inner.rotary_emb.to(embed_home)
        if hasattr(inner, "norm") and inner.norm is not None:
            inner.norm.to(exit_home)
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            try:
                self.model.lm_head.to(exit_home)
            except Exception:
                pass

        total_offloaded = 0
        placement_counts: dict[int, int] = {}
        slice_bytes = 0

        for idx, layer in enumerate(layers):
            home_gpu = (0 if idx < half else (n_gpus - 1)) if split else 0
            home = f"cuda:{home_gpu}"
            self._home_gpu[idx] = home_gpu
            self._init_gpu_cache(home_gpu)
            placement_counts[home_gpu] = placement_counts.get(home_gpu, 0) + 1
            layer.to(home)

            if split:
                bridge = layer.register_forward_pre_hook(
                    self._make_layer_bridge(home), with_kwargs=True,
                )
                self._handles.append(bridge)

            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is None or not hasattr(experts, "gate_up_proj"):
                continue

            mod_bytes = 0
            for p in experts.parameters():
                total_offloaded += p.numel() * p.element_size()
                mod_bytes += p.numel() * p.element_size()
                self._pin_cpu(p)

            mod_id = id(experts)
            self._module_size[mod_id] = mod_bytes
            self._module_by_id[mod_id] = experts
            self._module_home_gpu[mod_id] = home_gpu

            # One expert slice size (gate_up row + down row) for logging.
            if slice_bytes == 0 and experts.gate_up_proj.shape[0] > 0:
                e0 = 0
                gu = experts.gate_up_proj.data[e0]
                dn = experts.down_proj.data[e0]
                slice_bytes = (gu.numel() + dn.numel()) * gu.element_size()

            if self._slice_mode:
                experts._qwen_offloader = self  # noqa: SLF001
                experts._qwen_home_gpu = home_gpu  # noqa: SLF001
                experts._qwen_orig_forward = experts.forward
                experts.forward = types.MethodType(_slice_experts_forward, experts)
            else:
                pre = experts.register_forward_pre_hook(
                    self._make_pre_hook(home, home_gpu),
                )
                post = experts.register_forward_hook(self._make_post_hook())
                self._handles.append(pre)
                self._handles.append(post)

        for gpu in placement_counts:
            torch.cuda.synchronize(gpu)

        self._installed = True
        cap_mib = self._cache_gib * 1024
        mode = f"split {placement_counts}" if split else "single cuda:0"
        paging = "slice-LRU" if self._slice_mode else "layer-LRU"
        if self._slice_mode and slice_bytes:
            n_fit = int(cap_mib * 1024**2 // slice_bytes)
            unit = f"~{n_fit} expert-slices/GPU"
        else:
            min_mod = min(self._module_size.values()) if self._module_size else 1
            n_fit = int(cap_mib // (min_mod / 1024**2))
            unit = f"~{n_fit} modules/GPU"
        print(f"[expert-offload] {n_layers} layers, mode={mode}, paging={paging}; "
              f"{total_offloaded/1024**3:.2f} GiB routed experts on CPU "
              f"(pinned DMA). Per-GPU cache: {cap_mib:.0f} MiB ({unit})",
              flush=True)

    # ---- slice LRU ---------------------------------------------------------

    def _touch_slice(self, key: tuple, gpu: int):
        order = self._slice_order[gpu]
        if key in order:
            order.remove(key)
        order.append(key)

    def _free_vram_bytes(self, gpu: int) -> float:
        try:
            free, _ = torch.cuda.mem_get_info(gpu)
            return free
        except Exception:
            return 0.0

    def _evict_slices_until_fits(self, need: int, gpu: int):
        safety = float(os.environ.get("DEMO_EXPERT_CACHE_SAFETY_GIB", "1.0")) * 1024**3
        order = self._slice_order[gpu]
        cache = self._slice_cache[gpu]
        while order:
            if self._free_vram_bytes(gpu) - need > safety:
                return
            victim = order.pop(0)
            entry = cache.pop(victim, None)
            if entry is not None:
                self._slice_used[gpu] -= entry[2]

    def get_expert_slices(self, experts: nn.Module, expert_idx: int, gpu: int):
        """Return (gate_up, down_proj) weight matrices on cuda:gpu (LRU)."""
        key = (id(experts), expert_idx)
        cache = self._slice_cache[gpu]
        if key in cache:
            gate_up, down, sz = cache[key]
            self._touch_slice(key, gpu)
            return gate_up, down

        gu_cpu = experts.gate_up_proj.data[expert_idx]
        dn_cpu = experts.down_proj.data[expert_idx]
        sz = (gu_cpu.numel() + dn_cpu.numel()) * gu_cpu.element_size()
        self._evict_slices_until_fits(sz, gpu)
        home = f"cuda:{gpu}"
        gate_up = gu_cpu.to(home, non_blocking=True)
        down = dn_cpu.to(home, non_blocking=True)
        cache[key] = (gate_up, down, sz)
        self._slice_order[gpu].append(key)
        self._slice_used[gpu] += sz
        return gate_up, down

    # ---- layer LRU (legacy) ------------------------------------------------

    def _touch(self, mod_id: int, gpu: int):
        order = self._cache_order[gpu]
        if mod_id in order:
            order.remove(mod_id)
        order.append(mod_id)

    def _evict_until_fits(self, need_bytes: int, gpu: int):
        safety = float(os.environ.get("DEMO_EXPERT_CACHE_SAFETY_GIB", "1.0")) * 1024**3
        order = self._cache_order[gpu]
        while order:
            if self._free_vram_bytes(gpu) - need_bytes > safety:
                return
            victim = order.pop(0)
            experts = self._module_by_id.get(victim)
            if experts is None:
                continue
            sz = self._module_size.get(victim, 0)
            for p in experts.parameters():
                if p.device.type == "cuda":
                    cpu = torch.empty(
                        p.data.shape, dtype=p.data.dtype,
                        device="cpu", pin_memory=True,
                    )
                    cpu.copy_(p.data)
                    p.data = cpu
            self._cache_used_bytes[gpu] -= sz
            self._on_gpu[gpu].discard(victim)

    def _make_pre_hook(self, home: str, home_gpu: int):
        def hook(_module, args):
            mod_id = id(_module)
            on_gpu = self._on_gpu[home_gpu]
            if mod_id in on_gpu or all(p.device.type == "cuda" for p in _module.parameters()):
                if mod_id not in on_gpu:
                    on_gpu.add(mod_id)
                    self._cache_used_bytes[home_gpu] += self._module_size.get(mod_id, 0)
                self._touch(mod_id, home_gpu)
            else:
                sz = self._module_size.get(mod_id, 0)
                self._evict_until_fits(sz, home_gpu)
                for p in _module.parameters():
                    if p.device.type == "cpu":
                        p.data = p.data.to(home, non_blocking=False)
                on_gpu.add(mod_id)
                self._cache_used_bytes[home_gpu] += sz
                self._touch(mod_id, home_gpu)
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and str(a.device) != home:
                    a = a.to(home, non_blocking=False)
                new_args.append(a)
            return tuple(new_args)

        return hook

    def _make_post_hook(self):
        def hook(_module, _args, output):
            return output

        return hook

    def _make_layer_bridge(self, home: str):
        def hook(_layer, args, kwargs):
            def _mv(x):
                if isinstance(x, torch.Tensor) and x.device.type == "cuda" \
                        and str(x.device) != home:
                    return x.to(home, non_blocking=True)
                if isinstance(x, (list, tuple)):
                    return type(x)(_mv(v) for v in x)
                return x

            new_args = tuple(_mv(a) for a in args)
            if kwargs:
                kwargs = {k: _mv(v) for k, v in kwargs.items()}
            return new_args, kwargs

        return hook


def _slice_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Qwen3MoeExperts.forward with per-expert slice LRU (weights stay on CPU)."""
    offloader: ExpertOffloader = self._qwen_offloader  # noqa: SLF001
    gpu: int = self._qwen_home_gpu  # noqa: SLF001
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_row in expert_hit:
        expert_idx = int(expert_idx_row[0])
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up, down = offloader.get_expert_slices(self, expert_idx, gpu)
        gate, up = F.linear(current_state, gate_up).chunk(2, dim=-1)
        current_hidden = self.act_fn(gate) * up
        current_hidden = F.linear(current_hidden, down)
        current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden.to(final_hidden_states.dtype),
        )
    return final_hidden_states