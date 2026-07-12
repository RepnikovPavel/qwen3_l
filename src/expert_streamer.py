"""
Expert-level offload for Qwen3-MoE across the available GPUs.

The model is 31 GB; one 16 GB GPU can't hold it, and loading it through a
device_map that mixes in CPU triggers transformers' FP8 "weights conversion"
validator (the checkpoint is ALREADY quantized — that path is wrong).

Strategy (the "flat layout + expert load/unload" you asked for):
  - Load via device_map="auto" across the GPUs ONLY (no CPU entry) → transformers
    accepts it without conversion and splits the layers flat across cards
    (layers 0..N/2 on gpu0, rest on gpu1). 31 GB / 2 ≈ 15.5 GB each — fits.
  - After load, walk every MoE layer and move ONLY its `mlp.experts` weight
    tensors (gate_up_proj, down_proj — 27 GB total) to CPU. The rest
    (attention, router, shared expert, layernorms, embeddings — ~2 GB per GPU)
    stays resident.
  - Install forward pre/post hooks on each `mlp.experts` module that page its
    stacked tensors onto the experts' home GPU right before they run, and back
    to CPU after. Only the active layer's experts (~576 MiB) are on a GPU at
    any moment, so VRAM stays ~1 GB resident + one expert slab.

We use module hooks (not a forward override) so the stock generate() works
unchanged — the experts module is called normally by the MoE block; our hook
just relocates its parameters in time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ExpertOffloader:
    """Keeps routed-expert weight tensors on CPU, pages them per forward."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list = []
        self._installed = False
        # Per-expert-module home GPU (discovered from where the module lives).
        self._home_gpu: dict[int, int] = {}

    @classmethod
    def install(cls, model: nn.Module) -> "ExpertOffloader":
        od = cls(model)
        od._install()
        return od

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False
        # Restore each experts module to its home GPU.
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
            # Qwen3MoeExperts has the stacked gate_up_proj / down_proj params.
            if experts is not None and hasattr(experts, "gate_up_proj"):
                yield experts

    def _install(self):
        if self._installed:
            return
        n = 0
        total_bytes = 0
        for idx, experts in enumerate(self._iter_expert_modules()):
            # Discover this module's home GPU from any of its params.
            try:
                sample = next(experts.parameters())
                home_gpu = sample.device.index if sample.device.type == "cuda" else 0
            except StopIteration:
                home_gpu = 0
            self._home_gpu[idx] = home_gpu
            home = f"cuda:{home_gpu}"

            # Move expert params to CPU now (resident state = offloaded).
            for p in experts.parameters():
                total_bytes += p.numel() * p.element_size()
                p.data = p.data.to("cpu", non_blocking=False)
            # Pre-hook: copy expert params onto their home GPU before the module runs.
            pre = experts.register_forward_pre_hook(self._make_pre_hook(home))
            # Post-hook: copy them back to CPU after (frees VRAM).
            post = experts.register_forward_hook(self._make_post_hook())
            self._handles.append(pre)
            self._handles.append(post)
            n += 1
        if torch.cuda.is_available():
            for g in range(torch.cuda.device_count()):
                torch.cuda.synchronize(f"cuda:{g}")
        self._installed = True
        print(f"[expert-offload] {n} expert modules moved to CPU "
              f"({total_bytes/1024**3:.2f} GiB); resident per-GPU = rest of model",
              flush=True)

    def _make_pre_hook(self, home: str):
        """Build a closure that pages the expert tensors onto their home GPU."""
        def hook(_module, args):
            for p in _module.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.to(home, non_blocking=False)
            # The activation tensor is produced by the MoE block (on the home GPU
            # already since router/shared live there), but be defensive.
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and str(a.device) != home:
                    a = a.to(home, non_blocking=False)
                new_args.append(a)
            return tuple(new_args)

        return hook

    def _make_post_hook(self):
        """Build a closure that pages expert tensors back to CPU after use."""
        def hook(_module, _args, output):
            for p in _module.parameters():
                if p.device.type == "cuda":
                    p.data = p.data.to("cpu", non_blocking=False)
            return output

        return hook

