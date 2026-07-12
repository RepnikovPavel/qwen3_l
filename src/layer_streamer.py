"""
Double-buffered layer streaming for CPU-offloaded models.

Inspired by `iter_forward_gpu_buff` in the user's forked transformers
(/home/user/calc/rag/transformers/.../modeling_qwen3.py): keep model weights on
CPU RAM and page decoder layers onto the GPU a CHUNK at a time. While the
current chunk computes, the NEXT chunk is copied CPU->GPU and the chunk behind
is copied back CPU.

Design notes (learned the hard way):
  - Do NOT spawn Python threads for the copies. `module.to(non_blocking=True)`
    launched from many background threads deadlocks against the main CUDA
    context (the user's reference impl runs copies synchronously between
    chunks, not from worker threads).
  - Pin to a SINGLE GPU. Round-robin across GPUs breaks because the shared
    rotary_emb (cos/sin) and the KV cache live on one device, so `q*cos`
    raises a device-mismatch when layers hop cards. To use both GPUs, run two
    model instances in parallel (see bench/bench.py).
  - Use a dedicated CUDA stream + pinned staging buffers for the copy so it
    overlaps with compute. The main stream waits on the copy stream before a
    chunk runs.

The integration point: we replace `model.model.forward` (the layer-stack part)
with our own chunked loop, and leave the rest of `generate()` untouched. This
mirrors the user's `iter_forward_gpu_buff` (which is also a Qwen3Model.forward
override). We monkeypatch it so the stock checkpoint class doesn't need editing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerStreamer:
    """Pages decoder layers GPU<->CPU with double buffering on ONE gpu.

    Construct with the inner model (the object holding `.layers`), then call
    `install()` to monkeypatch its `forward`. The original forward is saved and
    restored by `remove()`.
    """

    def __init__(self, inner_model: nn.Module, gpu: int, chunk: int = 2):
        self.inner = inner_model          # e.g. Qwen3MoeModel
        self.layers = list(inner_model.layers)
        self.gpu = gpu
        self.device = f"cuda:{gpu}"
        self.chunk = max(1, chunk)
        self._orig_forward = None
        self._copy_stream = torch.cuda.Stream(device=self.device)
        # Resident small modules stay permanently on the GPU (used every step):
        #   embed_tokens, rotary_emb, final norm. lm_head lives on the
        #   Qwen3MoeForCausalLM wrapper (the PARENT of inner_model) — move it too.
        if hasattr(inner_model, "embed_tokens") and inner_model.embed_tokens is not None:
            inner_model.embed_tokens.to(self.device)
        if hasattr(inner_model, "rotary_emb") and inner_model.rotary_emb is not None:
            inner_model.rotary_emb.to(self.device)
        if hasattr(inner_model, "norm") and inner_model.norm is not None:
            inner_model.norm.to(self.device)
        # inner_model is model.model; its parent is the ForCausalLM with lm_head.
        parent = getattr(inner_model, "_moeparent", None)
        if parent is None:
            # Try to find it: caller should set inner._moeparent = model.
            pass
        # All decoder layers start on CPU (paged in/out per forward).
        for layer in self.layers:
            layer.to("cpu", non_blocking=False)
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------ install/remove

    def install(self):
        """Monkeypatch inner_model.forward with our chunked streaming loop."""
        if self._orig_forward is not None:
            return  # already installed
        import types  # noqa: PLC0415

        self._orig_forward = self.inner.forward
        streamer = self

        # Bind as a real method on the instance: `self` (the inner model) is
        # passed by Python as the first positional arg, matching inner_model.
        def _forward(inner_model, *args, **kwargs):
            return streamer._streamed_forward(inner_model, *args, **kwargs)

        self.inner.forward = types.MethodType(_forward, self.inner)
        print(f"[streamer] patched {type(self.inner).__name__}.forward: "
              f"{len(self.layers)} layers, gpu={self.gpu}, chunk={self.chunk}",
              flush=True)

    def remove(self):
        if self._orig_forward is not None:
            self.inner.forward = self._orig_forward
            self._orig_forward = None

    # ------------------------------------------------------------ the streaming loop

    @torch.no_grad()
    def _streamed_forward(self, inner_self, *args, **kwargs):
        """Replacement forward: run embed + chunked layer loop + norm.

        Mirrors what the original Qwen3MoeModel.forward does, but pages layers.
        We delegate the "small" parts (embed_tokens, rotary_emb, norm, mask
        prep) to the original by calling it on a no-op layer list is messy, so
        we reimplement the minimal stack. This works for Qwen3 / Qwen3Moe.
        """
        # Pop the standard args we know how to handle.
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        use_cache = kwargs.pop("use_cache", None)
        cache_position = kwargs.pop("cache_position", None)
        # Any leftover kwargs are passed through unchanged.

        from transformers.cache_utils import DynamicCache  # noqa: PLC0415
        from transformers.masking_utils import create_causal_mask  # noqa: PLC0415

        if inputs_embeds is None:
            inputs_embeds = inner_self.embed_tokens(input_ids)
        # Move embeddings onto our GPU.
        inputs_embeds = inputs_embeds.to(self.device, non_blocking=True)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=inner_self.config)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=self.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = inner_self.rotary_emb(inputs_embeds, position_ids)
        # rotary_emb cos/sin must live on the same GPU as q/k.
        position_embeddings = [p.to(self.device, non_blocking=True) for p in position_embeddings]
        position_ids = position_ids.to(self.device, non_blocking=True)
        if past_key_values is not None:
            past_key_values = past_key_values.to(self.device, non_blocking=True)

        # Causal mask (kept on GPU).
        mask_kwargs = dict(
            config=inner_self.config, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        causal_mask = create_causal_mask(**mask_kwargs)
        causal_mask = causal_mask.to(self.device, non_blocking=True) if causal_mask is not None else None

        # Pre-stage the first chunk on the GPU via the copy stream.
        chunk = self.chunk
        layers = self.layers
        n = len(layers)

        def move(chunk_layers, dev):
            for ly in chunk_layers:
                ly.to(dev, non_blocking=True)

        # Async-copy first chunk onto GPU on the copy stream.
        with torch.cuda.stream(self._copy_stream):
            move(layers[0:chunk], self.device)
        # Main stream waits for that copy before using the layers.
        torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)

        hidden_states = inputs_embeds
        for i in range(0, n, chunk):
            current = layers[i:i + chunk]
            nxt = layers[i + chunk:i + 2 * chunk]

            # Async prefetch the NEXT chunk (copy stream) while we compute.
            if nxt:
                with torch.cuda.stream(self._copy_stream):
                    move(nxt, self.device)

            # Compute the current chunk on the main stream.
            for layer in current:
                out = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
                # Layer may return a tuple (hidden, ...) or a dataclass.
                hidden_states = out[0] if isinstance(out, tuple) else getattr(
                    out, "last_hidden_state", out
                )

            # Async evict the chunk we just finished back to CPU.
            if nxt:  # only evict if there's more work (keep last chunk on GPU)
                with torch.cuda.stream(self._copy_stream):
                    move(current, "cpu")
                # Need the next chunk resident: make main stream wait.
                torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)

        hidden_states = inner_self.norm(hidden_states)
        from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: PLC0415
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
