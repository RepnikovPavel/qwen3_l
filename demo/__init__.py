"""Streaming inference helpers for the web demo.

Re-exports ModelInfo from src.model_info (single source of truth for the KV
budget) and provides the TextIteratorStreamer wrapper + the thinking/answer
splitter used by the SSE endpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.model_info import ModelInfo  # noqa: F401  (re-export)
from transformers import TextIteratorStreamer


@dataclass
class GenerationStats:
    """Live counters updated during streaming and emitted to the client."""

    n_new_tokens: int = 0
    first_token_time: float | None = None
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
        """Decode throughput over generated tokens (excluding prefill)."""
        if self.first_token_time is None or self.last_token_time is None:
            return 0.0
        span = self.last_token_time - self.first_token_time
        if span <= 0:
            return 0.0
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
    if saw_think_end:
        return "", text
    return text, ""


def fmt_bytes(n: int | float) -> str:
    from src.human_size import fmt_bytes as _fmt  # noqa: PLC0415

    return _fmt(n)
