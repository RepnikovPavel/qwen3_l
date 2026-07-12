"""
Expert-level offload for Qwen3-MoE across the available GPUs.

The model is 31 GB. transformers REJECTS loading an FP8 checkpoint through a
device_map that mixes GPU+CPU ("attempting to load an FP8 model with a
device_map that contains a cpu/disk device — not supported when the model is
quantized on the fly"). And it doesn't fit on a single 16 GB GPU either.

So we load to CPU only (device_map="cpu" — no validator trigger, no conversion),
then MANUALLY place modules:
  - routed experts (the big stacked tensors, 27 GB)  -> stay on CPU, paged per call
  - everything else (attention, router, shared expert, embed, norm, lm_head)
    -> split flat across the GPUs (~1 GB per card)

With 2 GPUs we use contiguous-half placement (layers 0..N/2 on cuda:0,
N/2..N on cuda:1) and bridge activations + rotary cos/sin via layer pre-hooks
(with_kwargs=True). Each GPU gets its own LRU expert cache (DEMO_EXPERT_CACHE_GIB
per card, default 11 GiB).

Hooks (not a forward override) page each experts module's tensors GPU<-CPU
right before it runs — stock generate() works unchanged.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class ExpertOffloader:
    """Keeps routed-expert weight tensors on CPU, pages them per forward.

    Call ``place_model`` on a CPU-loaded model to distribute the non-expert
    modules across GPUs, then the experts auto-page via forward hooks.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list = []
        self._installed = False
        self._home_gpu: dict[int, int] = {}  # layer idx -> gpu id
        # Per-GPU LRU cache state
        self._cache_gib = float(os.environ.get("DEMO_EXPERT_CACHE_GIB", "11"))
        self._cache_used_bytes: dict[int, int] = {}   # gpu -> bytes
        self._cache_order: dict[int, list[int]] = {}  # gpu -> module ids LRU
        self._on_gpu: dict[int, set[int]] = {}        # gpu -> module ids resident
        self._module_size: dict[int, int] = {}        # module id -> bytes
        self._module_by_id: dict[int, nn.Module] = {}
        self._module_home_gpu: dict[int, int] = {}    # module id -> home gpu
        self._split_gpus = False

    @classmethod
    def install(cls, model: nn.Module) -> "ExpertOffloader":
        """Place non-expert modules across GPUs + install paging hooks."""
        od = cls(model)
        od._place_and_hook()
        return od

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False
        for idx, experts in enumerate(self._iter_expert_modules()):
            gpu = self._home_gpu.get(idx)
            if gpu is not None:
                try:
                    experts.to(f"cuda:{gpu}")
                except Exception:
                    pass

    def _iter_expert_modules(self):
        """Yield every Qwen3MoeExperts module (one per MoE layer)."""
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

        if split:
            embed_home = "cuda:0"
            exit_home = f"cuda:{n_gpus - 1}"  # norm/lm_head follow last layers
        else:
            embed_home = exit_home = "cuda:0"

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

        for idx, layer in enumerate(layers):
            if split:
                home_gpu = 0 if idx < half else (n_gpus - 1)
            else:
                home_gpu = 0
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
            if experts is not None and hasattr(experts, "gate_up_proj"):
                mod_bytes = 0
                for p in experts.parameters():
                    total_offloaded += p.numel() * p.element_size()
                    mod_bytes += p.numel() * p.element_size()
                    p.data = p.data.to("cpu", non_blocking=False)
                    try:
                        p.data = torch.empty(
                            p.data.shape, dtype=p.data.dtype,
                            device="cpu", pin_memory=True,
                        ).copy_(p.data)
                    except RuntimeError:
                        pass
                mod_id = id(experts)
                self._module_size[mod_id] = mod_bytes
                self._module_by_id[mod_id] = experts
                self._module_home_gpu[mod_id] = home_gpu

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
        min_mod = min(self._module_size.values()) if self._module_size else 1
        n_fit = int(cap_mib // (min_mod / 1024**2))
        mode = (f"split {placement_counts}" if split
                else "single cuda:0")
        print(f"[expert-offload] {n_layers} layers, mode={mode}; "
              f"{total_offloaded/1024**3:.2f} GiB routed experts on CPU "
              f"(pinned DMA). Per-GPU cache: {cap_mib:.0f} MiB "
              f"(~{n_fit} modules/GPU after warmup)", flush=True)

    def _touch(self, mod_id: int, gpu: int):
        order = self._cache_order[gpu]
        if mod_id in order:
            order.remove(mod_id)
        order.append(mod_id)

    def _free_vram_bytes(self, gpu: int) -> float:
        try:
            free, _total = torch.cuda.mem_get_info(gpu)
            return free
        except Exception:
            return 0.0

    def _evict_until_fits(self, need_bytes: int, gpu: int):
        safety_gib = float(os.environ.get("DEMO_EXPERT_CACHE_SAFETY_GIB", "1.0"))
        hard_floor = safety_gib * 1024**3
        order = self._cache_order[gpu]
        while order:
            free = self._free_vram_bytes(gpu)
            if free - need_bytes > hard_floor:
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
            already_on_gpu = all(p.device.type == "cuda"
                                 for p in _module.parameters())
            if mod_id in on_gpu or already_on_gpu:
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
        """Move activations + rotary cos/sin onto this layer's home GPU."""
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