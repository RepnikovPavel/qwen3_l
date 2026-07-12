"""
Registry of supported models.

Each entry collects everything the loader/UI/bench need to know about a model,
so adding a new checkpoint is a one-place change. The repo currently ships two:

  - Qwen3-4B-Thinking-2507-FP8  : dense, fits a single 16 GB GPU
  - Qwen3-30B-A3B-Thinking-2507-FP8 : MoE, 31 GB, needs model-parallel or
    CPU-expert offload to run on 2x16 GB

The "id" is the short key used in URLs / UI selects; the rest mirrors the HF
layout under $CKPTDIR/models--<slug>/snapshots/main.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str               # short id used in the UI / API
    hf_repo: str          # HuggingFace repo (also the cache slug)
    family: str           # "qwen3" or "qwen3_moe"
    display_name: str     # human-friendly
    fp8: bool             # FP8 weights (needs the local Triton kernel on GPU)
    moe: bool             # Mixture-of-Experts
    # Approximate size on disk (bytes) — used for the UI/download hints.
    size_bytes: int
    short_desc: str

    @property
    def cache_dir(self) -> str:
        return f"models--{self.hf_repo.replace('/', '--')}/snapshots/main"

    def path(self, ckptdir: str) -> str:
        import os
        return os.path.join(ckptdir, self.cache_dir)


MODELS: dict[str, ModelSpec] = {
    "qwen3-4b": ModelSpec(
        id="qwen3-4b",
        hf_repo="Qwen/Qwen3-4B-Thinking-2507-FP8",
        family="qwen3",
        display_name="Qwen3-4B-Thinking-2507-FP8 (dense)",
        fp8=True,
        moe=False,
        size_bytes=5_190_053_264,
        short_desc="4B dense, FP8. Fits one 16 GB GPU (≈4.5 GB VRAM).",
    ),
    "qwen3-30b-a3b": ModelSpec(
        id="qwen3-30b-a3b",
        hf_repo="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        family="qwen3_moe",
        display_name="Qwen3-30B-A3B-Thinking-2507-FP8 (MoE)",
        fp8=True,
        moe=True,
        # 4 shards: 10 + 10 + 10 + 1.17 GB
        size_bytes=31_200_000_000,
        short_desc="30B MoE (128 experts, 8 active), FP8. 31 GB — needs "
                   "model-parallel and/or CPU expert offload on 2x16 GB.",
    ),
}


def get_model(model_id: str) -> ModelSpec:
    if model_id not in MODELS:
        raise KeyError(
            f"unknown model id {model_id!r}; known: {list(MODELS)}"
        )
    return MODELS[model_id]


def list_models() -> list[ModelSpec]:
    return [MODELS[k] for k in ("qwen3-4b", "qwen3-30b-a3b")]
