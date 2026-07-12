"""Streaming inference helpers for the web demo.

Wraps model.generate with a TextIteratorStreamer so tokens flow out as they're
produced. While streaming we measure decode throughput (tokens/s) and classify
each chunk as 'thinking' (before </think>) or 'answer' (after), so the UI can
render the two panels live.

KV-cache budget is computed from the model config so the UI can show how much
of the available context has been consumed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from transformers import TextIteratorStreamer

# </think> token id in the Qwen3 tokenizer.
THINK_END_ID = 151668


@dataclass
class ModelInfo:
    """Static model facts used for the context budget computation."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    dtype_bytes: int = 2  # bf16

    @classmethod
    def from_model(cls, model) -> "ModelInfo":
        cfg = model.config
        # head_dim is explicit on Qwen3; fall back to hidden/num_heads.
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        dtype = next(model.parameters()).dtype
        # bf16/fp16 = 2 bytes, fp32 = 4.
        dtype_bytes = 2 if dtype.itemsize <= 2 else 4
        return cls(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", 32768),
            dtype_bytes=dtype_bytes,
        )

    @property
    def bytes_per_token(self) -> int:
        """KV-cache bytes per token: K+V across all layers & KV heads."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def max_kv_bytes(self) -> int:
        return self.bytes_per_token * self.max_position_embeddings


@dataclass
class GenerationStats:
    """Live counters updated during streaming and emitted to the client."""

    n_new_tokens: int = 0
    first_token_time: float | None = None  # time of first generated token
    last_token_time: float | None = None
    prompt_tokens: int = 0
    started_at: float = field(default_factory=time.time)

    def on_token(self):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.n_new_tokens += 1

    @property
    def tokens_per_s(self) -> float:
        """Decode throughput over the generated tokens (excluding prefill).

        Measured from the first generated token to the last, so the prefill
        latency doesn't drag down the reported decode speed.
        """
        if self.first_token_time is None or self.last_token_time is None:
            return 0.0
        span = self.last_token_time - self.first_token_time
        if span <= 0:
            return 0.0
        # n_new_tokens-1 intervals between n_new_tokens tokens.
        intervals = max(self.n_new_tokens - 1, 1)
        return intervals / span


def make_streamer(tokenizer) -> TextIteratorStreamer:
    return TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,  # keep </think> so we can split panels
    )


def split_thinking(text: str, saw_think_end: bool) -> tuple[str, str]:
    """Split a partial decode into (thinking, answer) on the last </think>."""
    if "</think>" in text:
        idx = text.rfind("</think>")
        thinking = text[:idx]
        answer = text[idx + len("</think>"):].lstrip("\n")
        return thinking, answer
    # No </think> yet: everything is thinking (unless we already passed it).
    if saw_think_end:
        return "", text
    return text, ""


def fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def context_used_tokens(model_info: ModelInfo, current_seq_len: int) -> tuple[int, int]:
    """Return (used_tokens, max_tokens) for the context budget.

    ``max_tokens`` is capped by the model's max_position_embeddings; the
    practical KV-cache ceiling is GPU memory, surfaced separately by the server.
    """
    return current_seq_len, model_info.max_position_embeddings
