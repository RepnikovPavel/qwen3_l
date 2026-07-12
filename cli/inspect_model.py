"""
qwen3_l CLI — model inspection + layer/transfer benchmarks.

Two subcommands:

  inspect  — static facts read from the model config + state-dict sizes:
               * max context (tokens), KV bytes/token, KV bytes vs context size
               * per-layer weight size, broken down by component (attention,
                 router, shared expert, routed experts for MoE)
               * total + active parameter counts
             No GPU needed (loads config + a size-only state dict).

  bench    — dynamic measurements (needs a GPU):
               * single-layer load GPU<-CPU and unload CPU<-GPU (ms)
               * single-layer forward pass (ms) vs context length, on GPU and CPU
               * KV-cache/context transfer time: GPU0<->GPU1, GPU->CPU (ms)
             All times are reported in milliseconds, averaged over a few runs.

Usage:
    python -m cli.inspect_model inspect --model qwen3-30b-a3b --ckptdir /path/to/hf
    python -m cli.inspect_model bench   --model qwen3-30b-a3b --ckptdir /path/to/hf
    python -m cli.inspect_model inspect --model qwen3-4b
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import get_model  # noqa: E402


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _fmt_num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ===========================================================================
# inspect — static facts
# ===========================================================================

def _config_for(model_id: str, ckptdir: str):
    """Load just the config (no weights) — cheap."""
    import json  # noqa: PLC0415
    spec = get_model(model_id)
    path = os.path.join(ckptdir, spec.cache_dir, "config.json")
    with open(path) as f:
        return json.load(f), spec


def _param_count_of_module(mod) -> tuple[int, int]:
    """Return (total_params, total_bytes) of a module."""
    total = 0
    bytes_ = 0
    for p in mod.parameters():
        total += p.numel()
        bytes_ += p.numel() * p.element_size()
    return total, bytes_


def cmd_inspect(args):
    cfg, spec = _config_for(args.model, args.ckptdir)
    is_moe = cfg.get("model_type", "").endswith("_moe") or "num_experts" in cfg

    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or hidden // n_heads
    max_pos = cfg.get("max_position_embeddings", 0)
    # bf16 by default.
    dtype_bytes = 2

    print("=" * 72)
    print(f"MODEL: {spec.display_name}")
    print(f"  family : {cfg.get('model_type')}  {'(MoE)' if is_moe else '(dense)'}")
    print(f"  layers : {n_layers}, hidden={hidden}, heads={n_heads} "
          f"(kv_heads={n_kv_heads}), head_dim={head_dim}")
    print(f"  dtype  : {cfg.get('torch_dtype', '?')}")
    if is_moe:
        ne = cfg.get("num_experts", 0)
        k = cfg.get("num_experts_per_tok", 0)
        mff = cfg.get("moe_intermediate_size", 0)
        sff = cfg.get("intermediate_size", 0)
        print(f"  experts: {ne} routed, top-{k} active, moe_ff={mff}, "
              f"shared_ff={sff}")
    print("=" * 72)

    # ----- Context / KV-cache budget ---------------------------------------
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    print("\n[CONTEXT]")
    print(f"  max context (tokens)        : {_fmt_num(max_pos)}")
    print(f"  KV-cache bytes/token        : {kv_per_token} ({_fmt_bytes(kv_per_token)})")
    print(f"  KV-cache at max context     : {_fmt_bytes(kv_per_token * max_pos)}")
    # Sample a few context sizes so the reader sees the growth.
    print("  KV-cache vs context size    :")
    for ctx in (512, 2048, 8192, 32768, 131072, max_pos):
        if ctx <= max_pos:
            mb = kv_per_token * ctx / 1024**2
            print(f"    {ctx:>7} tokens -> {mb:>10.1f} MiB")

    # ----- Weight sizes per component --------------------------------------
    print("\n[WEIGHTS per decoder layer]")
    _print_layer_breakdown(cfg, is_moe)

    # ----- Totals ----------------------------------------------------------
    print("\n[TOTALS]")
    _print_totals(cfg, n_layers, hidden, is_moe, dtype_bytes, max_pos, kv_per_token)


def _print_layer_breakdown(cfg, is_moe):
    """Compute weight sizes per layer component from config dims."""
    hidden = cfg["hidden_size"]
    n_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim") or hidden // cfg["num_attention_heads"]
    # QKV sizes: q is hidden*hidden, k/v are hidden*(kv_heads*head_dim).
    q_dim = hidden
    kv_dim = n_kv_heads * head_dim
    # bf16 dense weights; FP8 where the model is FP8 (1 byte).
    fp8 = cfg.get("quantization_config", {}).get("quant_method") == "fp8"
    fp8_b = 1 if fp8 else 2
    # Attention: q, k, v, o projections (no bias).
    attn_bytes = (hidden * q_dim + 2 * hidden * kv_dim + hidden * hidden) * fp8_b
    print(f"  attention (Q,K,V,O)         : {_fmt_bytes(attn_bytes)}  [{fp8_b}B/param]")
    # layernorms (2 * hidden, fp32 usually but small).
    ln_bytes = 2 * hidden * 4
    print(f"  layernorms (2x RMSNorm)     : {_fmt_bytes(ln_bytes)}")
    if is_moe:
        ne = cfg["num_experts"]
        k = cfg["num_experts_per_tok"]
        mff = cfg["moe_intermediate_size"]
        sff = cfg["intermediate_size"]
        # Routed experts: stacked tensors gate_up (ne, 2*mff, hidden) + down (ne, hidden, mff).
        routed_bytes = ne * (2 * mff * hidden + hidden * mff) * fp8_b
        # Shared expert: gate_up (2*sff, hidden) + down (hidden, sff).
        shared_bytes = (2 * sff * hidden + hidden * sff) * fp8_b
        # Router weight: (ne, hidden).
        router_bytes = ne * hidden * 2
        print(f"  router (gate)               : {_fmt_bytes(router_bytes)}")
        print(f"  shared expert               : {_fmt_bytes(shared_bytes)}")
        print(f"  routed experts ({ne}×)       : {_fmt_bytes(routed_bytes)}")
        active_routed = k * (2 * mff * hidden + hidden * mff) * fp8_b
        print(f"    └ active/token (top-{k})    : {_fmt_bytes(active_routed)}")
        print(f"  ─────────────────────────────")
        print(f"  layer total                 : "
              f"{_fmt_bytes(attn_bytes + ln_bytes + router_bytes + shared_bytes + routed_bytes)}")
    else:
        ff = cfg.get("intermediate_size", 4 * hidden)
        ffn_bytes = (2 * ff * hidden + hidden * ff) * fp8_b
        print(f"  FFN (gate,up,down)          : {_fmt_bytes(ffn_bytes)}")
        print(f"  ─────────────────────────────")
        print(f"  layer total                 : "
              f"{_fmt_bytes(attn_bytes + ln_bytes + ffn_bytes)}")


def _print_totals(cfg, n_layers, hidden, is_moe, dtype_bytes, max_pos, kv_per_token):
    """Compute grand totals: total params, active params, weights bytes."""
    fp8 = cfg.get("quantization_config", {}).get("quant_method") == "fp8"
    fp8_b = 1 if fp8 else 2
    # Re-use per-layer numbers.
    n_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim") or hidden // cfg["num_attention_heads"]
    q_dim = hidden
    kv_dim = n_kv_heads * head_dim
    attn_bytes = (hidden * q_dim + 2 * hidden * kv_dim + hidden * hidden) * fp8_b
    embed_bytes = cfg["vocab_size"] * hidden * fp8_b

    if is_moe:
        ne = cfg["num_experts"]
        k = cfg["num_experts_per_tok"]
        mff = cfg["moe_intermediate_size"]
        sff = cfg["intermediate_size"]
        routed = ne * (2 * mff * hidden + hidden * mff) * fp8_b
        shared = (2 * sff * hidden + hidden * sff) * fp8_b
        router = ne * hidden * 2
        per_layer = attn_bytes + 2 * hidden * 4 + router + shared + routed
        active_per_layer = attn_bytes + k * (2 * mff * hidden + hidden * mff) * fp8_b \
            + (2 * sff * hidden + hidden * sff) * fp8_b
    else:
        ff = cfg.get("intermediate_size", 4 * hidden)
        ffn = (2 * ff * hidden + hidden * ff) * fp8_b
        per_layer = attn_bytes + 2 * hidden * 4 + ffn
        active_per_layer = per_layer

    total_weights = per_layer * n_layers + embed_bytes
    active_weights = active_per_layer * n_layers + embed_bytes
    print(f"  total weight bytes          : {_fmt_bytes(total_weights)}")
    if is_moe:
        print(f"  active weight bytes/token   : {_fmt_bytes(active_weights)}")
        print(f"  sparsity (inactive/total)   : "
              f"{(1 - active_weights/total_weights)*100:.1f}%")
    print(f"  KV-cache @ max context      : {_fmt_bytes(kv_per_token * max_pos)}")


# ===========================================================================
# bench — dynamic measurements (needs GPU)
# ===========================================================================

def cmd_bench(args):
    import torch  # noqa: PLC0415
    from src.inject_kernel import inject_fp8_kernel  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    if not torch.cuda.is_available():
        sys.exit("bench requires CUDA (no GPU found)")

    spec = get_model(args.model)
    inject_fp8_kernel()
    cfg = AutoConfig.from_pretrained(spec.path(args.ckptdir), local_files_only=True)
    path = spec.path(args.ckptdir)
    n_gpus = torch.cuda.device_count()
    print("=" * 72)
    print(f"BENCH: {spec.display_name}  ({n_gpus} GPU(s))")
    print("=" * 72)

    # Load ONE decoder layer to a known state for the per-layer timings.
    print("\n[loading full model to CPU for layer isolation...]")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", device_map="cpu",
        local_files_only=True, attn_implementation="sdpa",
    ).eval()
    layers = list(model.model.layers)
    sample_layer = layers[0]

    # ---- 1) single-layer load / unload ------------------------------------
    _bench_layer_transfer(sample_layer, args.runs)

    # ---- 2) layer forward pass vs context length --------------------------
    _bench_layer_forward(sample_layer, model, cfg, args)

    # ---- 3) context (KV-cache) transfers ----------------------------------
    _bench_context_transfer(model, cfg, args)


def _bench_layer_transfer(layer, runs: int):
    import torch  # noqa: PLC0415
    print("\n[SINGLE-LAYER TRANSFER]  (avg over runs)")
    layer.to("cpu")
    torch.cuda.synchronize()
    size_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    print(f"  layer weight size: {_fmt_bytes(size_bytes)}")

    dev = "cuda:0"
    # Warm up.
    layer.to(dev); layer.to("cpu"); torch.cuda.synchronize()

    # Load CPU -> GPU.
    t0 = time.perf_counter()
    for _ in range(runs):
        layer.to(dev)
        torch.cuda.synchronize()
        layer.to("cpu")
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"  load   CPU -> GPU0 : {load_ms:8.2f} ms/run")

    # Unload GPU -> CPU (measured separately).
    layer.to(dev); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        layer.to("cpu")
        torch.cuda.synchronize()
        layer.to(dev)
        torch.cuda.synchronize()
    unload_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"  unload GPU0 -> CPU : {unload_ms:8.2f} ms/run")


def _bench_layer_forward(layer, model, cfg, args):
    """Time a single decoder layer's forward pass at several context lengths."""
    import torch  # noqa: PLC0415
    dev = "cuda:0"
    cpu_dev = "cpu"
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads

    print("\n[LAYER FORWARD PASS]  (ms, single layer, batch=1)")

    def make_inputs(ctx_len, device):
        # hidden states, position_ids, position_embeddings (cos,sin), causal mask
        hs = torch.randn(1, ctx_len, hidden, dtype=torch.bfloat16, device=device)
        pos = torch.arange(ctx_len, device=device).unsqueeze(0)
        # rotary cos/sin: call rotary_emb
        return hs, pos

    rotary = getattr(model.model, "rotary_emb", None)
    lengths = [l for l in (1, 128, 512, 2048, 8192) if l <= args.max_ctx]
    print(f"  {'ctx':>6} | {'GPU ms':>10} | {'CPU ms':>10}")
    print("  " + "-" * 38)

    for ctx in lengths:
        # --- GPU ---
        try:
            layer.to(dev); torch.cuda.synchronize()
            hs, pos = make_inputs(ctx, dev)
            pos_emb = rotary(hs, pos) if rotary else None
            if pos_emb:
                pos_emb = [p.to(dev) for p in pos_emb]
            with torch.no_grad():
                for _ in range(args.warmup):  # warmup
                    layer(hs, position_ids=pos, position_embeddings=pos_emb)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.runs):
                    layer(hs, position_ids=pos, position_embeddings=pos_emb)
                torch.cuda.synchronize()
                gpu_ms = (time.perf_counter() - t0) / args.runs * 1000
        except torch.cuda.OutOfMemoryError:
            gpu_ms = float("nan")
        finally:
            layer.to("cpu"); torch.cuda.empty_cache()

        # --- CPU ---
        try:
            layer.to(cpu_dev)
            hs, pos = make_inputs(ctx, cpu_dev)
            pos_emb = rotary(hs, pos) if rotary else None
            with torch.no_grad():
                for _ in range(max(1, args.warmup // 4)):
                    layer(hs, position_ids=pos, position_embeddings=pos_emb)
                t0 = time.perf_counter()
                for _ in range(max(1, args.runs // 4)):
                    layer(hs, position_ids=pos, position_embeddings=pos_emb)
                cpu_ms = (time.perf_counter() - t0) / max(1, args.runs // 4) * 1000
        except Exception as e:
            cpu_ms = float("nan")

        print(f"  {ctx:>6} | {gpu_ms:>10.2f} | {cpu_ms:>10.2f}")


def _bench_context_transfer(model, cfg, args):
    """Time KV-cache / activation tensor transfers.

    'context' here = a batch of (K,V) tensors of the full model at a given
    sequence length — the dominant cost when moving state between devices.
    """
    import torch  # noqa: PLC0415
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # KV cache shape: 2 (k,v) * n_layers * (batch, n_kv, ctx, head_dim) bf16.
    print("\n[CONTEXT (KV-cache) TRANSFER]  (ms)")
    print(f"  KV-cache shape: 2 * {n_layers} layers * "
          f"(1, {n_kv}, ctx, {head_dim}) bf16")
    lengths = [l for l in (512, 2048, 8192) if l <= args.max_ctx]
    n_gpus = torch.cuda.device_count()
    has_2gpu = n_gpus >= 2

    print(f"  {'ctx':>6} | {'KV MiB':>9} | {'GPU0->CPU':>10}"
          + (f" | {'GPU0->GPU1':>11}" if has_2gpu else "")
          + f" | {'CPU->GPU0':>10}")
    print("  " + "-" * (60 if has_2gpu else 48))

    for ctx in lengths:
        # Build a representative KV blob (one contiguous-ish set of tensors).
        per_layer = 2 * n_kv * head_dim * ctx * 2  # bytes (k+v, bf16)
        total_bytes = n_layers * per_layer
        # Allocate as a single tensor for transfer timing (approximates cost).
        blob_gpu0 = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
        torch.cuda.synchronize()

        # GPU0 -> CPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            cpu_blob = blob_gpu0.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        g2c_ms = (time.perf_counter() - t0) / args.runs * 1000

        # GPU0 -> GPU1
        g2g_ms = float("nan")
        if has_2gpu:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.runs):
                g1_blob = blob_gpu0.to("cuda:1", non_blocking=False)
            torch.cuda.synchronize()
            g2g_ms = (time.perf_counter() - t0) / args.runs * 1000

        # CPU -> GPU0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            blob_gpu0.copy_(cpu_blob, non_blocking=False)
        torch.cuda.synchronize()
        c2g_ms = (time.perf_counter() - t0) / args.runs * 1000

        mib = total_bytes / 1024**2
        row = (f"  {ctx:>6} | {mib:>9.1f} | {g2c_ms:>10.2f}")
        if has_2gpu:
            row += f" | {g2g_ms:>11.2f}"
        row += f" | {c2g_ms:>10.2f}"
        print(row)

    del blob_gpu0
    try:
        del cpu_blob
    except NameError:
        pass
    torch.cuda.empty_cache()


# ===========================================================================
# main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(prog="cli.inspect_model",
                                description="qwen3_l model inspection + benchmarks")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="static model facts (no GPU needed)")
    pi.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-30b-a3b"])
    pi.add_argument("--ckptdir", default=os.environ.get("CKPTDIR", "/mnt/nvme/huggingface"))
    pi.set_defaults(func=cmd_inspect)

    pb = sub.add_parser("bench", help="dynamic transfer/forward benchmarks (GPU)")
    pb.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-30b-a3b"])
    pb.add_argument("--ckptdir", default=os.environ.get("CKPTDIR", "/mnt/nvme/huggingface"))
    pb.add_argument("--runs", type=int, default=5, help="iterations to average")
    pb.add_argument("--warmup", type=int, default=3, help="warmup iterations (forward)")
    pb.add_argument("--max-ctx", type=int, default=8192,
                    help="max context length to benchmark")
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
