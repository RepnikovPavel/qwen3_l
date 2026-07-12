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
        # Decide a home GPU for each layer: split in CONTIGUOUS halves
        # (layers 0..N/2-1 on gpu0, N/2..N-1 on gpu1) — minimizes cross-GPU
        # boundary crossings vs round-robin, and matches what device_map="auto"
        # would do. The non-expert parts move to the home GPU; routed experts
        # stay on CPU and page to it.
        inner = getattr(self.model, "model", self.model)
        layers = list(getattr(inner, "layers", []))
        n_layers = len(layers)
        split = (n_layers + n_gpus - 1) // n_gpus  # ceil — first half slightly bigger

        # Entry modules (embed_tokens) on GPU 0.
        if hasattr(inner, "embed_tokens") and inner.embed_tokens is not None:
            inner.embed_tokens.to("cuda:0")
        # rotary_emb shared across layers — keep on GPU 0 (layers are called in
        # order; the activation hops to each layer's home GPU via the hook).
        if hasattr(inner, "rotary_emb") and inner.rotary_emb is not None:
            inner.rotary_emb.to("cuda:0")
        # Exit modules (final norm, lm_head) on the last GPU.
        last_gpu = n_gpus - 1
        if hasattr(inner, "norm") and inner.norm is not None:
            inner.norm.to(f"cuda:{last_gpu}")
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            try:
                self.model.lm_head.to(f"cuda:{last_gpu}")
            except Exception:
                pass

        total_offloaded = 0
        for idx, layer in enumerate(layers):
            gpu = min(idx // split, n_gpus - 1)
            home = f"cuda:{gpu}"
            self._home_gpu[idx] = gpu
            # Move everything in this layer to the home GPU first...
            layer.to(home)
            # ...then push the routed-expert tensors back to CPU.
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is not None and hasattr(experts, "gate_up_proj"):
                for p in experts.parameters():
                    total_offloaded += p.numel() * p.element_size()
                    p.data = p.data.to("cpu", non_blocking=False)
                # Install paging hooks on this experts module.
                pre = experts.register_forward_pre_hook(self._make_pre_hook(home))
                post = experts.register_forward_hook(self._make_post_hook())
                self._handles.append(pre)
                self._handles.append(post)
            # Install a pre-hook on the whole layer to bridge activations onto
            # its home GPU (needed at the GPU0/GPU1 boundary; also moves the
            # rotary cos/sin kwarg along with the activation).
            layer.register_forward_pre_hook(self._make_layer_bridge(home),
                                            with_kwargs=True)

        for g in range(n_gpus):
            torch.cuda.synchronize(f"cuda:{g}")
        self._installed = True
        print(f"[expert-offload] placed {n_layers} layers across {n_gpus} GPU(s); "
              f"{total_offloaded/1024**3:.2f} GiB of routed experts on CPU",
              flush=True)

    def _make_pre_hook(self, home: str):
        """Page the expert tensors onto their home GPU before the module runs."""
        def hook(_module, args):
            for p in _module.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.to(home, non_blocking=False)
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and str(a.device) != home:
                    a = a.to(home, non_blocking=False)
                new_args.append(a)
            return tuple(new_args)

        return hook

    def _make_post_hook(self):
        """Page expert tensors back to CPU after use."""
        def hook(_module, _args, output):
            for p in _module.parameters():
                if p.device.type == "cuda":
                    p.data = p.data.to("cpu", non_blocking=False)
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


