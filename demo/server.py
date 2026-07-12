"""
FastAPI web demo server for Qwen3-4B-Thinking-2507-FP8.

Runs the model on the server (model-parallel across all visible GPUs by default
— layers split flat across cards so a single model spans them), and streams
generated tokens to a browser UI via Server-Sent Events.

Layout of the response stream (one JSON object per line, SSE `data:` frames):
  {"type":"meta",   ...}                — session info: model, gpu placement
  {"type":"prompt", "tokens": N}        — prompt token count (prefill size)
  {"type":"ctx",    "used":..,"max":.., "kv_bytes":..} — context budget
  {"type":"token",  "phase":"thinking"|"answer", "text": ".."} — per-token
  {"type":"stats",  "tps":.., "n":.., "elapsed":..}   — final statistics
  {"type":"done"}
  {"type":"error",  "message": ".."}

Access from your laptop via an SSH tunnel (the server binds to 0.0.0.0 so the
tunnel can reach it):

    ssh -L 8000:localhost:8000 tuna-server
    # then open http://localhost:8000 in your browser
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# Offline mode must be set before transformers is imported (it's imported via
# src.inference). Set it early, right here.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inject_kernel import inject_fp8_kernel  # noqa: E402
from src.inference import _model_path  # noqa: E402
from demo import (  # noqa: E402
    GenerationStats,
    ModelInfo,
    context_used_tokens,
    fmt_bytes,
    make_streamer,
    split_thinking,
)

app = FastAPI(title="Qwen3-4B-Thinking-2507-FP8 demo")

# ----------------------------------------------------------------------------
# Globals: model + tokenizer loaded once at import (kept hot across requests).
# ----------------------------------------------------------------------------
STATE: dict = {"tokenizer": None, "model": None, "info": None, "placement": None}


def _load():
    if STATE["model"] is not None:
        return
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    inject_fp8_kernel()
    ckptdir = os.environ.get("CKPTDIR", "/mnt/nvme/huggingface")
    model_path = _model_path(ckptdir)
    print(f"[demo] loading model from {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    n_gpus = torch.cuda.device_count()
    device = os.environ.get("DEMO_DEVICE", "mp")  # default: model-parallel ("flat")
    if device == "mp" and n_gpus >= 2:
        # Model-parallel ("flat"): one model, layers split across all GPUs via
        # device_map="auto". This is the default scenario per the task.
        max_memory = None
        mm = os.environ.get("DEMO_MP_MAX_MEMORY")
        if mm:
            max_memory = {}
            for part in mm.split(","):
                dev, mem = part.split(":")
                max_memory[int(dev)] = mem
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        placement = dict(getattr(model, "hf_device_map", {}))
        print(f"[demo] model-parallel across {n_gpus} GPUs: "
              f"{len(placement)} module(s) placed", flush=True)
    elif device == "cpu":
        # CPU path: dequantize FP8 -> bf16 so no Triton kernel is needed.
        from src.dequant import dequantize_model  # noqa: PLC0415

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="cpu",
            local_files_only=True, attn_implementation="eager",
        ).eval()
        dequantize_model(model)
        model = model.to(torch.bfloat16)
        placement = {"cpu": "all"}
    else:
        # Single GPU (cuda:<id>).
        gpu_id = int(os.environ.get("DEMO_GPU_ID", "0"))
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="cpu",
            local_files_only=True, attn_implementation="sdpa",
        ).eval().to(f"cuda:{gpu_id}")
        placement = {f"cuda:{gpu_id}": "all"}

    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    STATE["info"] = ModelInfo.from_model(model)
    STATE["placement"] = placement
    print(f"[demo] ready. KV budget: {fmt_bytes(STATE['info'].max_kv_bytes)} "
          f"@ {STATE['info'].max_position_embeddings} tokens", flush=True)


# Eagerly load on import so the worker is hot as soon as it serves.
_load()


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 8192
    history: list[dict] | None = None  # [{"role":"user"/"assistant","content":".."}]


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")


@app.get("/api/info")
def info():
    """Static session info for the UI header."""
    i = STATE["info"]
    return {
        "model": "Qwen3-4B-Thinking-2507-FP8",
        "mode": os.environ.get("DEMO_DEVICE", "mp"),
        "placement": _summarize_placement(STATE["placement"]),
        "max_position_embeddings": i.max_position_embeddings,
        "kv_bytes_per_token": i.bytes_per_token,
        "kv_bytes_per_token_hr": fmt_bytes(i.bytes_per_token),
    }


def _summarize_placement(placement: dict) -> str:
    """Collapse the per-module device map to per-GPU layer ranges."""
    if not placement:
        return "unknown"
    if "cpu" in placement:
        return "CPU"
    # Group transformer layers by device.
    per_gpu: dict[int, list[int]] = {}
    for name, dev in placement.items():
        if ".layers." in name and dev in (0, 1) or isinstance(dev, int):
            try:
                li = int(name.split(".layers.")[1].split(".")[0])
                per_gpu.setdefault(int(dev), []).append(li)
            except (IndexError, ValueError):
                pass
    if not per_gpu:
        # Fallback: just list devices.
        devs = sorted(set(int(d) for d in placement.values() if isinstance(d, int)))
        return ", ".join(f"GPU {d}" for d in devs) if devs else str(placement)
    parts = []
    for dev in sorted(per_gpu):
        layers = sorted(per_gpu[dev])
        parts.append(f"GPU {dev}: layers {layers[0]}-{layers[-1]} ({len(layers)})")
    return " | ".join(parts)


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Generate a streamed response. Tokens flow as SSE data frames."""
    import torch  # noqa: PLC0415

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    info = STATE["info"]

    # Build the message list from history + the new turn.
    messages = list(req.history or [])
    messages.append({"role": "user", "content": req.message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            # Inputs go to the first layer's device (model-parallel) or model.device.
            if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None):
                input_device = next(iter(model.hf_device_map.values()))
            else:
                input_device = model.device
            inputs = tokenizer([text], return_tensors="pt").to(input_device)
            n_prompt = inputs.input_ids.shape[1]

            yield sse({"type": "prompt", "tokens": n_prompt})
            used, mx = context_used_tokens(info, n_prompt)
            yield sse({"type": "ctx", "used": used, "max": mx,
                       "kv_bytes": info.bytes_per_token * n_prompt,
                       "kv_bytes_hr": fmt_bytes(info.bytes_per_token * n_prompt),
                       "kv_bytes_per_token": info.bytes_per_token})

            streamer = make_streamer(tokenizer)
            stats = GenerationStats(prompt_tokens=n_prompt)

            # Run generation on a background thread; consume the streamer here.
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            t_start = time.perf_counter()
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            accumulated = ""
            saw_think_end = False
            last_thinking = ""
            last_answer = ""
            try:
                for piece in streamer:
                    accumulated += piece
                    stats.on_token()
                    thinking, answer = split_thinking(accumulated, saw_think_end)
                    if "</think>" in piece:
                        saw_think_end = True
                    # Emit only the delta for each panel.
                    if thinking != last_thinking:
                        yield sse({"type": "token", "phase": "thinking",
                                   "text": thinking[len(last_thinking):]})
                        last_thinking = thinking
                    if answer != last_answer:
                        yield sse({"type": "token", "phase": "answer",
                                   "text": answer[len(last_answer):]})
                        last_answer = answer
                    # Live stats + context (sequence grows by 1 per token).
                    seq_len = n_prompt + stats.n_new_tokens
                    yield sse({"type": "stats_live", "tps": round(stats.tokens_per_s, 2),
                               "n": stats.n_new_tokens})
                    yield sse({"type": "ctx", "used": seq_len, "max": mx,
                               "kv_bytes": info.bytes_per_token * seq_len,
                               "kv_bytes_hr": fmt_bytes(info.bytes_per_token * seq_len),
                               "kv_bytes_per_token": info.bytes_per_token})
            finally:
                thread.join()

            elapsed = time.perf_counter() - t_start
            yield sse({"type": "stats", "tps": round(stats.tokens_per_s, 2),
                       "n": stats.n_new_tokens, "elapsed": round(elapsed, 2),
                       "prompt_tokens": n_prompt})
            yield sse({"type": "done"})
        except Exception as e:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("demo.server:app", host="0.0.0.0", port=port, log_level="info")
