"""
Qwen3-4B-Thinking-2507-FP8 inference + throughput benchmark.

This is a cleaned-up, reusable version of the original demo. It:
  1. Forces offline mode (no Hub access).
  2. Injects the vendored finegrained-fp8 Triton kernel into transformers.
  3. Loads Qwen3-4B-Thinking-2507-FP8 from a local HF-cache snapshot.
  4. Runs generation and measures prompt / decode throughput in tokens/s.

Usage:
    python -m src.inference --device cuda          # GPU
    python -m src.inference --device cpu           # CPU (loads on CPU, no CUDA)
    python -m src.inference --ckptdir /path/to/hf  # custom checkpoint root
"""
from __future__ import annotations

import argparse
import os
import time

# ----------------------------------------------------------------------------
# 1. Force offline mode BEFORE importing transformers — no Hub downloads, ever.
#    Must run at import time so any later `import transformers` picks it up.
# ----------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ----------------------------------------------------------------------------
# 2. Inject the local FP8 kernel into transformers' kernel cache. Done lazily
#    inside run() so --help doesn't require the kernel / torch to be present.
# ----------------------------------------------------------------------------

MODEL_REL = "models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/main"

# </think> token id in the Qwen3 tokenizer; used to split thinking vs answer.
THINK_END_ID = 151668

DEFAULT_PROMPT = "Give me a short introduction to large language model."


def _model_path(ckptdir: str) -> str:
    return os.path.join(ckptdir, MODEL_REL)


def load_model(ckptdir: str, device: str):
    """Load tokenizer + model onto ``device``.

    Weights are loaded to CPU first (handles the large FP8 checkpoint without
    OOM on the GPU), then moved to the target device — matching the original
    demo's pattern. For ``device == "cpu"`` the model simply stays on CPU.
    """
    # Import here so the offline env vars above are already set.
    from src.inject_kernel import inject_fp8_kernel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # Register the local kernel before the model touches FP8 layers.
    inject_fp8_kernel()

    model_path = _model_path(ckptdir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True,
    ).eval()

    if device != "cpu":
        model = model.to(device)
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 32768, device: str = "cuda"):
    """Run one generation and return (output_ids, thinking, content, timing).

    Timing is split into prefill (processing the prompt) and decode (generating
    new tokens), so throughput is reported for each phase.
    """
    import torch  # noqa: PLC0415

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    n_prompt = model_inputs.input_ids.shape[1]

    with torch.no_grad():
        t0 = time.perf_counter()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()

    output_ids = generated_ids[0][n_prompt:].tolist()

    # Split thinking vs. final answer on the last </think>.
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    n_new = len(output_ids)
    elapsed = t1 - t0
    # Prefill vs decode can't be cleanly separated from outside model.generate
    # without hooks, so report end-to-end throughput over the generated tokens.
    tps = n_new / elapsed if elapsed > 0 else float("inf")

    return {
        "output_ids": output_ids,
        "thinking_content": thinking_content,
        "content": content,
        "n_prompt": n_prompt,
        "n_new": n_new,
        "elapsed_s": elapsed,
        "tokens_per_s": tps,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking-2507-FP8 inference / benchmark")
    parser.add_argument("--ckptdir", type=str, default="/mnt/nvme/huggingface",
                        help="HF cache root containing models--Qwen--Qwen3-4B-Thinking-2507-FP8")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Run on GPU (cuda) or CPU")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    args = parser.parse_args()

    print(f"[load] device={args.device} ckptdir={args.ckptdir}")
    tokenizer, model = load_model(args.ckptdir, args.device)

    print(f"[generate] prompt={args.prompt!r} max_new_tokens={args.max_new_tokens}")
    result = generate(tokenizer, model, args.prompt, args.max_new_tokens, args.device)

    print("=" * 60)
    print(f"prompt tokens : {result['n_prompt']}")
    print(f"new tokens    : {result['n_new']}")
    print(f"elapsed       : {result['elapsed_s']:.2f} s")
    print(f"throughput    : {result['tokens_per_s']:.2f} tokens/s  ({args.device})")
    print("-" * 60)
    print("thinking content:", result["thinking_content"])
    print("content:", result["content"])
    print("=" * 60)


if __name__ == "__main__":
    main()
