"""
Qwen3-4B-Thinking-2507-FP8 inference + throughput benchmark.

Supports three execution paths:
  - GPU (cuda): FP8 weights run via the vendored finegrained-fp8 Triton kernel
    (injected into transformers' kernel cache). Pin a specific GPU with
    --gpu-id, or run "all" to launch one process per GPU in parallel.
  - CPU: FP8 weights are dequantized to bf16 first (see src/dequant.py), then
    the model runs purely on the host CPU. No CUDA / Triton needed.

For the CPU↔GPU consistency check, generation is made deterministic via a fixed
seed + greedy decoding, so identical prompts produce identical token streams.

Usage:
    python -m src.inference --device cuda                 # GPU 0
    python -m src.inference --device cuda --gpu-id 1      # GPU 1
    python -m src.inference --device cpu                  # CPU (dequantized)
"""
from __future__ import annotations

import argparse
import os
import time

# ----------------------------------------------------------------------------
# 1. Force offline mode BEFORE importing transformers — no Hub downloads, ever.
# ----------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_REL = "models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/main"
THINK_END_ID = 151668
DEFAULT_PROMPT = "Give me a short introduction to large language model."


def _model_path(ckptdir: str) -> str:
    return os.path.join(ckptdir, MODEL_REL)


def load_model(ckptdir: str, device: str, gpu_id: int = 0, eager_attn: bool = False,
               max_memory: dict | None = None):
    """Load tokenizer + model for ``device``.

    device options:
      - "cuda": single-GPU. Loads to CPU then moves to cuda:<gpu_id>.
      - "mp":   model-parallel across multiple GPUs via device_map="auto"
                (one logical model split across cards → fits a larger context /
                larger KV cache than a single card).
      - "cpu":  dequantizes FP8 → bf16 layers so no Triton kernel is needed.

    For "mp", ``max_memory`` can cap per-device memory (e.g.
    {0: "14GiB", 1: "14GiB"}) to leave headroom for the KV cache; if omitted,
    accelerate balances the split automatically.

    Returns (tokenizer, model).
    """
    from src.inject_kernel import inject_fp8_kernel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # Register the local FP8 kernel before the model touches FP8 layers.
    # On CPU this kernel is never invoked (we dequantize first), but registering
    # it is harmless and keeps the load path identical.
    inject_fp8_kernel()

    model_path = _model_path(ckptdir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    attn_impl = "eager" if eager_attn else "sdpa"

    if device == "mp":
        # Model-parallel: place one model across multiple GPUs at load time.
        # We must NOT move it afterwards (accelerate owns the placement).
        import torch  # noqa: PLC0415

        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            raise RuntimeError(
                f"model-parallel needs >=2 GPUs, found {n_gpus}. Use --device cuda."
            )
        device_map_arg = "auto" if max_memory is None else "balanced"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map_arg,
            max_memory=max_memory,
            local_files_only=True,
            attn_implementation=attn_impl,
        ).eval()
        placement = getattr(model, "hf_device_map", {})
        print(f"[load] model-parallel across {n_gpus} GPUs, placement={placement}", flush=True)
        return tokenizer, model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True,
        attn_implementation=attn_impl,
    ).eval()

    if device == "cpu":
        # Replace FP8 layers with dequantized bf16 linears.
        import torch  # noqa: PLC0415

        from src.dequant import dequantize_model  # noqa: PLC0415

        n = dequantize_model(model)
        print(f"[load] dequantized {n} FP8 layers → bf16 for CPU", flush=True)
        model = model.to(torch.bfloat16)
    else:
        import torch  # noqa: PLC0415

        target = f"cuda:{gpu_id}"
        model = model.to(target)
        print(f"[load] model on {target} ({torch.cuda.get_device_name(gpu_id)})", flush=True)

    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 2048,
             device: str = "cuda", seed: int = 0):
    """Run one generation. Deterministic (greedy) when ``seed`` is set.

    Returns a dict with the output ids (for consistency checks), decoded
    thinking/answer content, token counts, and end-to-end throughput.
    """
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # For model-parallel, place inputs on the device of the first layer
    # (model.hf_device_map maps modules → devices; the embed/lm_head device is
    # where inputs must live). For single-device models, use model.device.
    if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None):
        hdm = model.hf_device_map
        input_device = next(iter(hdm.values())) if hdm else model.device
    else:
        input_device = model.device
    model_inputs = tokenizer([text], return_tensors="pt").to(input_device)
    n_prompt = model_inputs.input_ids.shape[1]

    with torch.no_grad():
        t0 = time.perf_counter()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy → deterministic for CPU↔GPU match
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        t1 = time.perf_counter()

    output_ids = generated_ids[0][n_prompt:].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    n_new = len(output_ids)
    elapsed = t1 - t0
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
    parser.add_argument("--ckptdir", type=str, default="/mnt/nvme/huggingface")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mp"],
                        help="cuda=single GPU, mp=model-parallel across GPUs, cpu=host")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index (single-GPU)")
    parser.add_argument("--max-memory", type=str, default=None,
                        help="per-GPU memory cap for model-parallel, e.g. '0:14GiB,1:14GiB'")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eager-attn", action="store_true", help="use eager attention (CPU-safe)")
    args = parser.parse_args()

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for part in args.max_memory.split(","):
            dev, mem = part.split(":")
            max_memory[int(dev)] = mem

    print(f"[load] device={args.device} gpu_id={args.gpu_id} ckptdir={args.ckptdir}", flush=True)
    tokenizer, model = load_model(args.ckptdir, args.device, args.gpu_id,
                                  args.eager_attn, max_memory=max_memory)

    print(f"[generate] prompt={args.prompt!r} max_new_tokens={args.max_new_tokens}", flush=True)
    result = generate(tokenizer, model, args.prompt, args.max_new_tokens, args.device, args.seed)

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
