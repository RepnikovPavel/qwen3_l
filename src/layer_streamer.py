"""
Double-buffered layer streaming for CPU-offloaded models.

Inspired by `iter_forward_gpu_buff` in the user's forked transformers
(/home/user/calc/rag/transformers/.../modeling_qwen3.py): keep the model
weights on CPU RAM, and page layers onto the GPU a CHUNK at a time. While the
current chunk computes, the NEXT chunk is copied CPU→GPU on a background thread
(non_blocking), and the previous chunk is copied back CPU (non_blocking). The
GPU therefore never stalls waiting on a copy.

The trick here vs the user's version: we can't replace `forward()` of a model
we load from a checkpoint (we don't control the class source), so instead we
install a forward PRE-HOOK on every decoder layer. Each layer, right before it
runs, ensures IT is on the right GPU and kicks off the async prefetch of the
layer N+`ahead` that will be needed soon. This integrates with the stock
`model.generate()` path — no custom forward loop.

Multi-GPU: layers are assigned to GPUs round-robin (0,1,0,1,...) so BOTH cards
stay busy. With chunked prefetch both GPUs are fed in parallel rather than one
waiting on the other at a hard boundary.

Usage (set on a model after loading it to CPU):
    LayerStreamer.install(model, devices=[0,1], chunk=2)
    ... model.generate(...)   # layers stream in/out automatically
    LayerStreamer.remove(model)
"""
from __future__ import annotations

import threading
import torch
import torch.nn as nn


def _move_async(layer: nn.Module, device):
    """Move a layer to `device` asynchronously (CUDA streams overlap copy+compute)."""
    layer.to(device, non_blocking=True)


class LayerStreamer:
    """Installs hooks that page decoder layers GPU<->CPU with double buffering.

    One instance per model. Assigns each decoder layer to a "home" GPU
    (round-robin across `devices`) and keeps the layer on CPU between forward
    calls. On each layer's forward it: (1) waits for that layer's prefetch to
    land on its home GPU, (2) starts prefetching the layer `ahead` steps later.
    """

    def __init__(self, layers: list[nn.Module], devices: list[int], chunk: int = 2):
        self.layers = layers
        self.devices = devices
        self.chunk = chunk
        # home_gpu[i] = which GPU layer i lives on when paged in.
        self.home_gpu = [devices[i % len(devices)] for i in range(len(layers))]
        self._hooks = []
        self._lock = threading.Lock()
        # Track the next-layer prefetch thread so we can join it before using it.
        self._prefetch_threads: dict[int, threading.Thread] = {}
        # Make sure all layers start on CPU (offload state).
        for layer in layers:
            layer.to("cpu", non_blocking=False)
        self._installed = False

    @classmethod
    def install(cls, model, devices: list[int] | None = None, chunk: int = 2) -> "LayerStreamer":
        """Find the decoder layers under model.model.layers and hook them.

        embed_tokens / lm_head / final norm are kept permanently on GPU 0 / the
        last GPU respectively (they're small and used every step).
        """
        import torch  # noqa: PLC0415

        n = torch.cuda.device_count()
        devices = devices or list(range(max(1, n)))
        layers = list(model.model.layers)
        streamer = cls(layers, devices, chunk)

        # Keep entry/exit small modules resident on the GPUs (not paged).
        if hasattr(model.model, "embed_tokens") and model.model.embed_tokens is not None:
            model.model.embed_tokens.to(f"cuda:{devices[0]}")
        last_gpu = devices[-1]
        if hasattr(model.model, "norm") and model.model.norm is not None:
            model.model.norm.to(f"cuda:{last_gpu}")
        if hasattr(model, "lm_head") and model.lm_head is not None:
            try:
                model.lm_head.to(f"cuda:{last_gpu}")
            except Exception:
                pass

        # Install a pre-hook on each layer that pages it in + prefetches ahead.
        for i, layer in enumerate(layers):
            hook = layer.register_forward_pre_hook(
                lambda _mod, _args, idx=i: streamer._page_in(idx)
            )
            streamer._hooks.append(hook)
        # Install post-hook on the LAST layer to kick off wrap-around prefetch
        # of layer 0 for the next token step (decode loop).
        streamer._installed = True
        # Prime the first chunk so the very first layer is already on GPU.
        streamer._prime()
        print(f"[streamer] installed on {len(layers)} layers, devices={devices}, "
              f"chunk={chunk}, home_gpu[0..]={streamer.home_gpu[:6]}...", flush=True)
        return streamer

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False

    # ----------------------------------------------------------- internals

    def _prime(self):
        """Prefetch the first `chunk` layers before generation starts."""
        for i in range(min(self.chunk, len(self.layers))):
            self._start_prefetch(i)

    def _start_prefetch(self, idx: int):
        """Kick off async CPU→home-GPU copy of layer idx (if not already running)."""
        if idx >= len(self.layers):
            return
        with self._lock:
            if idx in self._prefetch_threads and self._prefetch_threads[idx].is_alive():
                return  # already in flight
            layer = self.layers[idx]
            gpu = self.home_gpu[idx]
            t = threading.Thread(target=_move_async, args=(layer, f"cuda:{gpu}"))
            self._prefetch_threads[idx] = t
            t.start()

    def _page_in(self, idx: int):
        """Called right before layer idx's forward: ensure it's on its home GPU,
        then start prefetching the next layers, and evict old ones back to CPU."""
        # 1. Wait for this layer's prefetch to finish (it was kicked off earlier).
        with self._lock:
            t = self._prefetch_threads.get(idx)
        if t is not None:
            t.join()
            with self._lock:
                self._prefetch_threads.pop(idx, None)
        # Make sure it really is on the home GPU (synchronous backstop).
        gpu = self.home_gpu[idx]
        layer = self.layers[idx]
        try:
            p0 = next(layer.parameters())
            if str(p0.device) != f"cuda:{gpu}":
                layer.to(f"cuda:{gpu}", non_blocking=False)
        except StopIteration:
            pass

        # 2. Prefetch the next `chunk` layers ahead of time.
        for k in range(1, self.chunk + 1):
            self._start_prefetch(idx + k)

        # 3. Evict layers we're done with (more than `chunk` behind) back to CPU.
        evict_idx = idx - self.chunk
        if evict_idx >= 0:
            old = self.layers[evict_idx]
            # non_blocking copy on a side CPU stream; no need to wait.
            try:
                p = next(old.parameters())
                if p.device.type == "cuda":
                    threading.Thread(
                        target=_move_async, args=(old, "cpu")
                    ).start()
            except StopIteration:
                pass

    def home_device_for_inputs(self):
        """Device where embed_tokens lives — where input_ids should be placed."""
        return f"cuda:{self.devices[0]}"
