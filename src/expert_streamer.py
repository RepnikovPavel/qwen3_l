"""
Expert-level offload for Qwen3-MoE on a single GPU.

The model is 31 GB; a single 16 GB GPU can't hold it. But only the routed
experts dominate (27 GB of stacked weight tensors). Everything else
(attention, router, shared expert, layernorms, embeddings) is ~2 GB and fits
trivially.

Strategy (the "flat layout + expert load/unload" you asked for):
  - Load the FP8 checkpoint STRAIGHT onto one GPU with device_map={"":gpu}.
    NO weight conversion — the checkpoint is already FP8, transformers loads
    it as-is. (The earlier "weights conversion" error came from mixing CPU
    into device_map, which triggers the FP8 quantizer validator.)
  - After load, walk every MoE layer and move ONLY its `mlp.experts` weight
    tensors (gate_up_proj, down_proj) to CPU. The rest stays on the GPU.
  - Install a forward pre/post hook on each `mlp.experts` module that pages
    its stacked tensors GPU<-CPU right before the experts run, and back to
    CPU right after. Only the active layer's experts are on the GPU at any
    moment (~576 MiB), so VRAM stays ~2 GB resident + one expert slab.

We monkeypatch the inner-model forward (not the layer forward) so the hook
fire is automatic — no need to replace generate(). The expert module is
called normally by Qwen3MoeSparseMoeBlock.forward; our hook just relocates
its parameters in time.

Usage:
    ExpertOffloader.install(model, gpu_id=0)
    ... model.generate(...)   # experts page in/out automatically
    ExpertOffloader.remove(model)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ExpertOffloader:
    """Keeps routed-expert weight tensors on CPU, pages them per forward."""

    def __init__(self, model: nn.Module, gpu: int = 0):
        self.model = model
        self.gpu = gpu
        self.device = f"cuda:{gpu}"
        self._handles: list = []
        self._installed = False

    @classmethod
    def install(cls, model: nn.Module, gpu: int = 0) -> "ExpertOffloader":
        od = cls(model, gpu)
        od._install()
        return od

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False
        # Restore experts to GPU so the model is in a clean state.
        for experts in self._iter_expert_modules():
            try:
                experts.to(self.device)
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
        for experts in self._iter_expert_modules():
            # Move expert params to CPU now (resident state = offloaded).
            for p in experts.parameters():
                total_bytes += p.numel() * p.element_size()
                p.data = p.data.to("cpu", non_blocking=False)
            # Pre-hook: copy expert params onto the GPU before the module runs.
            pre = experts.register_forward_pre_hook(self._make_pre_hook())
            # Post-hook: copy them back to CPU after (frees VRAM).
            post = experts.register_forward_hook(self._make_post_hook())
            self._handles.append(pre)
            self._handles.append(post)
            n += 1
        torch.cuda.synchronize(self.device)
        self._installed = True
        print(f"[expert-offload] {n} expert modules on CPU "
              f"({total_bytes/1024**3:.2f} GiB); resident on GPU = rest of model",
              flush=True)

    def _make_pre_hook(self):
        """Build a closure that pages the expert tensors onto the GPU."""
        dev = self.device

        def hook(_module, args):
            # The MoE block calls experts(...) with the routed token batch.
            # Move BOTH the expert params AND the incoming activation tensor
            # onto the GPU. (The activation arrives on the GPU already since
            # attention/router/shared live there, but be defensive.)
            for p in _module.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.to(dev, non_blocking=False)
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.device.type != dev:
                    a = a.to(dev, non_blocking=False)
                new_args.append(a)
            # Return the (possibly relocated) args so the module sees them.
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
