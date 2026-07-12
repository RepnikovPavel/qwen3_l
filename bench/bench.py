"""
Run the Qwen3-4B-Thinking-2507-FP8 benchmark on GPU and/or CPU and print a
results table suitable for pasting into the README.

Each row reports the throughput (tokens/s over the generated tokens) for one
device. GPU uses CUDA; CPU runs the model purely on the host CPU.

Usage:
    python bench/bench.py --ckptdir /path/to/hf --devices cuda cpu
    python bench/bench.py --ckptdir /path/to/hf --devices cuda            # GPU only
    python bench/bench.py --ckptdir /path/to/hf --devices cpu --max_new_tokens 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when running bench/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import DEFAULT_PROMPT, generate, load_model  # noqa: E402


def bench_one(ckptdir: str, device: str, prompt: str, max_new_tokens: int) -> dict:
    print(f"\n=== {device.upper()} ===", flush=True)
    tokenizer, model = load_model(ckptdir, device)
    print(f"[generate] prompt={prompt!r} max_new_tokens={max_new_tokens}", flush=True)
    res = generate(tokenizer, model, prompt, max_new_tokens, device)
    print(
        f"new tokens={res['n_new']}  elapsed={res['elapsed_s']:.2f}s  "
        f"throughput={res['tokens_per_s']:.2f} tokens/s",
        flush=True,
    )
    # Free memory before the next device.
    del model
    try:
        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {
        "device": device,
        "tokens_per_s": round(res["tokens_per_s"], 2),
        "new_tokens": res["n_new"],
        "elapsed_s": round(res["elapsed_s"], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking-2507-FP8 benchmark")
    parser.add_argument("--ckptdir", type=str, default="/mnt/nvme/huggingface")
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda", "cpu"],
                        choices=["cuda", "cpu"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--model_version", type=str, default="Qwen3-4B-Thinking-2507-FP8")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    results = []
    for device in args.devices:
        results.append(bench_one(args.ckptdir, device, args.prompt, args.max_new_tokens))

    # Markdown table — the only thing the README is supposed to contain.
    print("\n" + "=" * 60)
    print("RESULTS (markdown):")
    print("model version | gpu token/s | cpu token/s")
    gpu = next((r["tokens_per_s"] for r in results if r["device"] == "cuda"), "-")
    cpu = next((r["tokens_per_s"] for r in results if r["device"] == "cpu"), "-")
    print(f"{args.model_version} | {gpu} | {cpu}")
    print("=" * 60)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model_version": args.model_version,
            "results": results,
        }, indent=2))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
