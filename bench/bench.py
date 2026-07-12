"""
Qwen3-4B-Thinking-2507-FP8 benchmark: parallel multi-GPU + CPU + consistency.

Design:
  - One worker PROCESS per GPU, launched simultaneously so every GPU is driven
    to 100% in parallel. Each worker loads its own model copy on cuda:<id>.
  - One CPU worker: dequantizes FP8 → bf16 and runs purely on the host CPU.
  - Consistency check: every worker runs the SAME prompt with greedy decoding
    (do_sample=False, fixed seed), so the output token ids must be identical
    across devices. We compare them pairwise and report match / mismatch.

CRITICAL: uses the 'spawn' multiprocessing start method. The default 'fork'
breaks CUDA ("Cannot re-initialize CUDA in forked subprocess"), and even the
main process must not touch torch.cuda before spawning — GPU discovery is done
inside the workers, and the main process only orchestrates.

Usage:
    python -m bench.bench --ckptdir /path/to/hf                    # auto: all GPUs + CPU
    python -m bench.bench --ckptdir /path/to/hf --no-cpu           # GPUs only
    python -m bench.bench --ckptdir /path/to/hf --gpus 0 1         # specific GPUs
    python -m bench.bench --ckptdir /path/to/hf --mp               # model-parallel
    python -m bench.bench --ckptdir /path/to/hf --max_new_tokens 2048
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

# 'spawn' MUST be set before any torch import anywhere — fork breaks CUDA.
mp.set_start_method("spawn", force=True)

# Make `src` importable when running bench/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _gpu_worker(gpu_id: int, ckptdir: str, prompt: str, max_new_tokens: int,
                seed: int, q):
    """Worker: load model on cuda:<gpu_id> and run one generation."""
    try:
        import torch  # noqa: PLC0415

        torch.cuda.set_device(gpu_id)
        from src.inference import generate, load_model  # noqa: PLC0415

        name = torch.cuda.get_device_name(gpu_id)
        cap = torch.cuda.get_device_capability(gpu_id)[0]
        print(f"[worker cuda:{gpu_id}] {name} (SM{cap})", flush=True)

        tokenizer, model = load_model(ckptdir, "cuda", gpu_id=gpu_id)
        res = generate(tokenizer, model, prompt, max_new_tokens, "cuda", seed)

        mem_peak_mb = torch.cuda.max_memory_allocated(gpu_id) / 1024 / 1024

        q.put({
            "worker": f"cuda:{gpu_id}",
            "device": "cuda",
            "gpu_id": gpu_id,
            "ok": True,
            "output_ids": res["output_ids"],
            "content": res["content"][:300],
            "tokens_per_s": round(res["tokens_per_s"], 2),
            "new_tokens": res["n_new"],
            "elapsed_s": round(res["elapsed_s"], 2),
            "mem_peak_mb": round(mem_peak_mb, 1),
            "gpu_name": name,
        })
    except Exception as e:  # noqa: BLE001
        q.put({"worker": f"cuda:{gpu_id}", "device": "cuda", "gpu_id": gpu_id,
               "ok": False, "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()})


def _mp_worker(ckptdir: str, prompt: str, max_new_tokens: int, seed: int,
               max_memory, q):
    """Worker: one model split across all visible GPUs (model-parallel).

    Trades per-token latency for a larger effective memory pool, so a bigger
    KV cache / longer context fits than on any single card.
    """
    try:
        import torch  # noqa: PLC0415

        from src.inference import generate, load_model  # noqa: PLC0415

        tokenizer, model = load_model(ckptdir, "mp", max_memory=max_memory)
        res = generate(tokenizer, model, prompt, max_new_tokens, "mp", seed)

        mem = {}
        for g in range(torch.cuda.device_count()):
            mem[g] = round(torch.cuda.max_memory_allocated(g) / 1024 / 1024, 1)

        q.put({
            "worker": "mp (all gpus)",
            "device": "mp",
            "ok": True,
            "output_ids": res["output_ids"],
            "content": res["content"][:300],
            "tokens_per_s": round(res["tokens_per_s"], 2),
            "new_tokens": res["n_new"],
            "elapsed_s": round(res["elapsed_s"], 2),
            "mem_peak_mb": mem,
            "n_gpus": torch.cuda.device_count(),
        })
    except Exception as e:  # noqa: BLE001
        q.put({"worker": "mp (all gpus)", "device": "mp", "ok": False,
               "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()})


def _cpu_worker(ckptdir: str, prompt: str, max_new_tokens: int, seed: int,
                n_threads: int, q):
    """Worker: dequantize FP8 → bf16 and run on CPU."""
    try:
        import torch  # noqa: PLC0415

        if n_threads:
            torch.set_num_threads(n_threads)

        from src.inference import generate, load_model  # noqa: PLC0415

        tokenizer, model = load_model(ckptdir, "cpu")
        res = generate(tokenizer, model, prompt, max_new_tokens, "cpu", seed)

        q.put({
            "worker": "cpu",
            "device": "cpu",
            "ok": True,
            "output_ids": res["output_ids"],
            "content": res["content"][:300],
            "tokens_per_s": round(res["tokens_per_s"], 2),
            "new_tokens": res["n_new"],
            "elapsed_s": round(res["elapsed_s"], 2),
            "num_threads": torch.get_num_threads(),
        })
    except Exception as e:  # noqa: BLE001
        q.put({"worker": "cpu", "device": "cpu", "ok": False,
               "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()})


def _discover_gpus(gpu_ids):
    """Discover GPUs WITHOUT initializing CUDA in the parent.

    nvidia-smi gives the count/names without touching the CUDA runtime, so the
    'spawn' workers can initialize CUDA cleanly. Falls back gracefully.
    """
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip().splitlines()
        all_gpus = []
        for line in out:
            idx, name, cap = [x.strip() for x in line.split(",")]
            all_gpus.append((int(idx), name, cap))
    except Exception as e:  # noqa: BLE001
        print(f"[plan] nvidia-smi unavailable ({e})", flush=True)
        return []

    if gpu_ids is None:
        chosen = all_gpus
    else:
        chosen = [g for g in all_gpus if g[0] in gpu_ids]
    return chosen


def _consistency_report(results: list[dict]) -> list[str]:
    """Compare output_ids across all successful workers, pairwise."""
    ok = [r for r in results if r.get("ok")]
    lines = []
    if len(ok) < 2:
        lines.append(f"only {len(ok)} successful worker(s) — no comparison possible")
        return lines

    ref = ok[0]
    ref_ids = ref["output_ids"]
    lines.append(f"reference: {ref['worker']} ({len(ref_ids)} tokens)")
    for r in ok[1:]:
        other = r["output_ids"]
        n = min(len(ref_ids), len(other))
        mismatches = [i for i in range(n) if ref_ids[i] != other[i]]
        status = "MATCH" if (len(ref_ids) == len(other) and not mismatches) else "MISMATCH"
        detail = "" if status == "MATCH" else (
            f" (len {len(other)} vs {len(ref_ids)}, first diff @ {mismatches[0] if mismatches else 'end'})"
        )
        lines.append(f"  {r['worker']}: {status}{detail}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking-2507-FP8 multi-GPU + CPU benchmark")
    parser.add_argument("--ckptdir", type=str, default="/mnt/nvme/huggingface")
    parser.add_argument("--gpus", type=int, nargs="*", default=None,
                        help="GPU ids to use in parallel mode (default: all visible GPUs)")
    parser.add_argument("--mp", action="store_true",
                        help="model-parallel: one model split across all GPUs (larger context). "
                             "Mutually exclusive with parallel single-GPU workers.")
    parser.add_argument("--mp-max-memory", type=str, default=None,
                        help="per-GPU memory cap for model-parallel, e.g. '0:14GiB,1:14GiB'")
    parser.add_argument("--no-cpu", action="store_true", help="skip the CPU worker")
    parser.add_argument("--no-gpu", action="store_true", help="skip GPU workers (CPU only)")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="torch CPU threads (default: torch's default)")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_version", type=str, default="Qwen3-4B-Thinking-2507-FP8")
    parser.add_argument("--out", type=str, default=None, help="path to write JSON results")
    args = parser.parse_args()

    # Discover GPUs via nvidia-smi (no CUDA init in parent → spawn-safe).
    discovered = [] if args.no_gpu else _discover_gpus(args.gpus)
    for idx, name, cap in discovered:
        print(f"[plan] GPU {idx}: {name} (SM{cap.split('.')[0]})", flush=True)
    gpu_ids = [g[0] for g in discovered]
    n_gpus = len(gpu_ids)

    use_mp = args.mp and n_gpus >= 2
    if args.mp and n_gpus < 2:
        print(f"[plan] --mp requested but only {n_gpus} GPU(s) found; falling back to parallel mode", flush=True)

    run_cpu = (not args.no_cpu)
    mode = "model-parallel" if use_mp else ("parallel" if gpu_ids else "none")
    print(f"[plan] mode={mode}  GPUs={gpu_ids}  CPU={run_cpu}  max_new_tokens={args.max_new_tokens}", flush=True)

    max_memory = None
    if use_mp and args.mp_max_memory:
        max_memory = {}
        for part in args.mp_max_memory.split(","):
            dev, mem = part.split(":")
            max_memory[int(dev)] = mem

    # Use a spawn-compatible Queue.
    q = mp.Queue()
    procs = []
    if use_mp:
        p = mp.Process(target=_mp_worker,
                       args=(args.ckptdir, args.prompt, args.max_new_tokens, args.seed,
                             max_memory, q))
        p.start()
        procs.append(p)
    else:
        for g in gpu_ids:
            p = mp.Process(target=_gpu_worker,
                           args=(g, args.ckptdir, args.prompt, args.max_new_tokens, args.seed, q))
            p.start()
            procs.append(p)
    if run_cpu:
        p = mp.Process(target=_cpu_worker,
                       args=(args.ckptdir, args.prompt, args.max_new_tokens, args.seed,
                             args.cpu_threads, q))
        p.start()
        procs.append(p)

    n_expected = len(procs)
    results = []
    for _ in range(n_expected):
        results.append(q.get())

    for p in procs:
        p.join()

    # ---- Report -----------------------------------------------------------
    print("\n" + "=" * 70)
    gpu_tps = []
    mp_tps = None
    cpu_tps = None
    for r in sorted(results, key=lambda x: (x["device"], x.get("gpu_id", -1))):
        if not r["ok"]:
            print(f"[FAIL] {r['worker']}: {r.get('error')}")
            if r.get("traceback"):
                print(r["traceback"])
            continue
        if r["device"] == "cuda":
            gpu_tps.append(r["tokens_per_s"])
            print(f"[OK] {r['worker']} ({r.get('gpu_name','?')}): {r['tokens_per_s']} tok/s | "
                  f"{r['new_tokens']} tok in {r['elapsed_s']}s | "
                  f"mem {r['mem_peak_mb']} MB")
        elif r["device"] == "mp":
            mp_tps = r["tokens_per_s"]
            print(f"[OK] {r['worker']}: {r['tokens_per_s']} tok/s | "
                  f"{r['new_tokens']} tok in {r['elapsed_s']}s | "
                  f"mem/GPU {r['mem_peak_mb']} MB")
        else:
            cpu_tps = r["tokens_per_s"]
            print(f"[OK] {r['worker']}: {r['tokens_per_s']} tok/s | "
                  f"{r['new_tokens']} tok in {r['elapsed_s']}s | "
                  f"{r['num_threads']} threads")

    # For the README table: aggregate per-GPU throughput.
    print("-" * 70)
    print("CPU↔GPU consistency check:")
    for line in _consistency_report(results):
        print("  " + line)

    print("\n" + "=" * 70)
    print("RESULTS (markdown):")
    print("model version | gpu token/s | cpu token/s | model-parallel token/s")
    gpu_str = ", ".join(f"{t}" for t in gpu_tps) if gpu_tps else "-"
    if len(gpu_tps) > 1:
        gpu_str = f"{gpu_str} (each), {round(sum(gpu_tps), 2)} total"
    cpu_str = cpu_tps if cpu_tps is not None else "-"
    mp_str = mp_tps if mp_tps is not None else "-"
    print(f"{args.model_version} | {gpu_str} | {cpu_str} | {mp_str}")
    print("=" * 70)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        slim = []
        for r in results:
            rr = {k: v for k, v in r.items() if k != "output_ids"}
            rr["output_ids_len"] = len(r.get("output_ids", []))
            slim.append(rr)
        Path(args.out).write_text(json.dumps({
            "model_version": args.model_version,
            "max_new_tokens": args.max_new_tokens,
            "prompt": args.prompt,
            "results": slim,
            "consistency": _consistency_report(results),
        }, indent=2))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
