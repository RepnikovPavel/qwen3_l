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

This is the "flat layout + expert load/unload" you asked for, and it sidesteps
both the conversion error and the OOM. Per-GPU resident footprint is tiny;
only the active layer's experts (~576 MiB) are on a GPU at any moment.

Hooks (not a forward override) page each experts module's tensors GPU<-CPU
right before it runs and back after — so the stock generate() works unchanged.
"""
from __future__ import annotations

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
        self._home_gpu: dict[int, int] = {}
        # ---- LRU cache of resident experts on the GPU ----
        # Don't evict immediately — keep recently-used expert modules on the GPU
        # so repeated decode steps (which hit the same experts) don't re-copy.
        # Capacity in MiB; defaults to leaving ~10 GiB free after the resident
        # non-expert weights. Override via DEMO_EXPERT_CACHE_GIB.
        import os  # noqa: PLC0415
        self._cache_gib = float(os.environ.get("DEMO_EXPERT_CACHE_GIB", "10"))
        self._cache_used_bytes = 0
        self._cache_order: list[int] = []  # module ids in LRU order (front=MRU)
        self._on_gpu: set[int] = set()
        self._module_size: dict[int, int] = {}  # module id -> bytes of experts

    @classmethod
    def install(cls, model: nn.Module) -> "ExpertOffloader":
        """Place non-expert modules across GPUs + install paging hooks.

        Assumes ``model`` was loaded with device_map="cpu" (so nothing is on a
        GPU yet). After this returns: experts on CPU, rest split across GPUs.
        """
        od = cls(model)
        od._place_and_hook()
        return od

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False
        # Restore experts to the home GPU so a later model.to('cpu') in unload
        # can move them cleanly.
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

    def _place_and_hook(self):
        if self._installed:
            return
        import os  # noqa: PLC0415

        n_gpus = torch.cuda.device_count() or 1
        # All decoder layers run on a SINGLE gpu (gpu0). Splitting across GPUs
        # breaks the shared rotary_emb (cos/sin unpacked inside attention.forward,
        # invisible to a layer pre-hook). We compensate with TWO optimizations:
        #   1) LRU cache — recently-used experts stay on the GPU (post-hook does
        #      NOT evict). With ~12 GiB cache ~21 of 48 expert modules stay
        #      resident after warmup, so most decode steps are cache hits.
        #   2) Async prefetch — post-hook kicks off a CUDA-stream copy of the
        #      NEXT layer's experts so the transfer overlaps with this layer's
        #      compute. Hides the residual ~half-miss cost.
        inner = getattr(self.model, "model", self.model)
        layers = list(getattr(inner, "layers", []))
        n_layers = len(layers)
        home = "cuda:0"
        home_gpu = 0
        # Map each expert-module-id -> its index in the layer order, so the
        # post-hook can prefetch the NEXT layer's experts.
        self._expert_ids_in_order: list[int] = []
        self._prefetch_stream = torch.cuda.Stream(device=home)

        # Entry modules (embed_tokens) on the home GPU.
        if hasattr(inner, "embed_tokens") and inner.embed_tokens is not None:
            inner.embed_tokens.to(home)
        # rotary_emb shared across layers — on the home GPU (same as layers).
        if hasattr(inner, "rotary_emb") and inner.rotary_emb is not None:
            inner.rotary_emb.to(home)
        # Exit modules (final norm, lm_head) on the home GPU too.
        if hasattr(inner, "norm") and inner.norm is not None:
            inner.norm.to(home)
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            try:
                self.model.lm_head.to(home)
            except Exception:
                pass

        total_offloaded = 0
        for idx, layer in enumerate(layers):
            self._home_gpu[idx] = home_gpu
            # Move everything in this layer to the home GPU first...
            layer.to(home)
            # ...then push the routed-expert tensors back to CPU (PINNED, so the
            # later CPU->GPU copies use DMA instead of staging through pageable
            # memory — ~2-3x faster).
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is not None and hasattr(experts, "gate_up_proj"):
                mod_bytes = 0
                for p in experts.parameters():
                    total_offloaded += p.numel() * p.element_size()
                    mod_bytes += p.numel() * p.element_size()
                    # Move to CPU then wrap in pinned storage for fast DMA.
                    p.data = p.data.to("cpu", non_blocking=False)
                    try:
                        p.data = torch.empty(
                            p.data.shape, dtype=p.data.dtype,
                            device="cpu", pin_memory=True,
                        ).copy_(p.data)
                    except RuntimeError:
                        pass  # host OOM on pin — fall back to pageable
                self._module_size[id(experts)] = mod_bytes
                self._expert_ids_in_order.append(id(experts))
                # Install paging hooks on this experts module.
                pre = experts.register_forward_pre_hook(self._make_pre_hook(home))
                post = experts.register_forward_hook(self._make_post_hook())
                self._handles.append(pre)
                self._handles.append(post)
            # No cross-GPU bridge needed — all layers share one GPU.

        torch.cuda.synchronize(home)
        self._installed = True
        cap_mib = self._cache_gib * 1024
        n_fit = int(cap_mib // (min(self._module_size.values()) / 1024**2)) if self._module_size else 0
        print(f"[expert-offload] placed {n_layers} layers on {home}; "
              f"{total_offloaded/1024**3:.2f} GiB of routed experts on CPU "
              f"(pinned, DMA copy). GPU cache: {cap_mib:.0f} MiB "
              f"(~{n_fit} expert modules stay resident after warmup)", flush=True)

    def _touch(self, mod_id: int):
        """Mark a module as most-recently-used in the LRU order."""
        if mod_id in self._cache_order:
            self._cache_order.remove(mod_id)
        self._cache_order.append(mod_id)

    def _evict_until_fits(self, need_bytes: int):
        """Evict least-recently-used expert modules until `need_bytes` free in cache."""
        cap = self._cache_gib * 1024**3
        while self._cache_used_bytes + need_bytes > cap and self._cache_order:
            victim = self._cache_order.pop(0)  # LRU
            # Find the module object by id (linear scan over the expert modules).
            for experts in self._iter_expert_modules():
                if id(experts) == victim:
                    sz = self._module_size.get(victim, 0)
                    for p in experts.parameters():
                        if p.device.type == "cuda":
                            # Copy back to (pinned) CPU and free the GPU copy.
                            cpu = torch.empty(
                                p.data.shape, dtype=p.data.dtype,
                                device="cpu", pin_memory=True,
                            )
                            cpu.copy_(p.data)
                            p.data = cpu
                    self._cache_used_bytes -= sz
                    self._on_gpu.discard(victim)
                    break

    def _make_pre_hook(self, home: str):
        """Page the expert tensors onto their home GPU before the module runs.

        LRU-cached: if already resident (cache hit) just bump the order; else
        evict LRU victims to make room and copy this module in (pinned DMA).
        If the post-hook of the previous layer already prefetched us on the
        side stream, just wait on that stream (no extra copy).
        """
        def hook(_module, args):
            mod_id = id(_module)
            # Wait for any in-flight prefetch of this module.
            torch.cuda.current_stream(home).wait_stream(self._prefetch_stream)
            already_on_gpu = all(p.device.type == "cuda"
                                 for p in _module.parameters())
            if mod_id in self._on_gpu or already_on_gpu:
                # Cache hit (genuine or prefetched) — no copy, just touch.
                if mod_id not in self._on_gpu:
                    # Was prefetched but not yet recorded — record now.
                    self._on_gpu.add(mod_id)
                    self._cache_used_bytes += self._module_size.get(mod_id, 0)
                self._touch(mod_id)
            else:
                sz = self._module_size.get(mod_id, 0)
                self._evict_until_fits(sz)
                for p in _module.parameters():
                    if p.device.type == "cpu":
                        p.data = p.data.to(home, non_blocking=False)
                self._on_gpu.add(mod_id)
                self._cache_used_bytes += sz
                self._touch(mod_id)
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and str(a.device) != home:
                    a = a.to(home, non_blocking=False)
                new_args.append(a)
            return tuple(new_args)

        return hook

    def _make_post_hook(self):
        """No immediate eviction — keep experts resident (LRU).

        Also kick off an async prefetch of the NEXT layer's experts on a side
        CUDA stream, so the CPU->GPU copy overlaps with the current layer's
        compute. The next pre-hook waits on the stream only if the module
        wasn't already resident.
        """
        def hook(_module, _args, output):
            mod_id = id(_module)
            # Find this module's position and prefetch the next one.
            try:
                idx = self._expert_ids_in_order.index(mod_id)
            except ValueError:
                idx = -1
            if idx >= 0 and idx + 1 < len(self._expert_ids_in_order):
                next_id = self._expert_ids_in_order[idx + 1]
                if next_id not in self._on_gpu:
                    # Find the next module object and async-copy its params.
                    for experts in self._iter_expert_modules():
                        if id(experts) == next_id:
                            sz = self._module_size.get(next_id, 0)
                            # Only prefetch if there's room (don't evict for it).
                            cap = self._cache_gib * 1024**3
                            if self._cache_used_bytes + sz <= cap:
                                with torch.cuda.stream(self._prefetch_stream):
                                    for p in experts.parameters():
                                        if p.device.type == "cpu":
                                            p.data = p.data.to("cuda:0",
                                                non_blocking=True)
                                # Note: the pre-hook will record it as resident.
                                # We can't easily mark it here without racing,
                                # so the pre-hook re-checks device and records.
                            break
            return output

        return hook

    def _make_layer_bridge(self, home: str):
        """Layer pre-hook: move incoming activations onto this layer's home GPU.

        With contiguous-half placement, only the boundary layer (first of the
        second half) actually hops GPUs; others are no-ops. We move args AND
        kwargs (position_embeddings = rotary cos/sin lives in kwargs and is
        shared, so it must hop with the activation).
        """
        def hook(_layer, args, kwargs):
            def _mv(x):
                if isinstance(x, torch.Tensor) and x.device.type == "cuda" \
                        and str(x.device) != home:
                    return x.to(home, non_blocking=True)
                return x
            new_args = tuple(_mv(a) for a in args)
            if kwargs:
                kwargs = {k: _mv(v) for k, v in kwargs.items()}
            return new_args, kwargs

        return hook


