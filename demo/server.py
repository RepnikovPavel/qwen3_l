"""
FastAPI web demo server for Qwen3 (4B dense + 30B MoE).

Features:
  - Model selection: qwen3-4b (dense) and qwen3-30b-a3b (MoE).
  - Lazy load: the model is loaded on the first /api/chat request and kept hot;
    auto-unloaded after DEMO_IDLE_TIMEOUT seconds of inactivity. Manual load /
    unload / reset-context endpoints exposed for the UI buttons.
  - Devices: mp (model-parallel, default), cuda (single GPU), cpu, and
    expert_offload for the 30B MoE (experts in CPU RAM).
  - Streams tokens via SSE, splitting each chunk into thinking/answer phases,
    with live tok/s and KV-cache context accounting.
  - /api/status reports per-GPU used/free MiB for the UI's VRAM meters.

Access from your laptop via an SSH tunnel:

    ssh -L 8000:localhost:8000 tuna-server
    # then open http://localhost:8000 in your browser
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_manager import ModelManager  # noqa: E402
from src.models import list_models  # noqa: E402
from demo import (  # noqa: E402
    GenerationStats,
    fmt_bytes,
    make_streamer,
    split_thinking,
)

app = FastAPI(title="Qwen3 demo (4B + 30B-MoE)")

CKPTDIR = os.environ.get("CKPTDIR", "/mnt/nvme/huggingface")
MANAGER = ModelManager(CKPTDIR)

# Per-model conversation history (kept here so reset_context can drop it).
HISTORY: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Static / metadata endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")


@app.get("/api/models")
def api_models():
    return {"models": [
        {"id": m.id, "display_name": m.display_name,
         "moe": m.moe, "fp8": m.fp8,
         "size_gb": round(m.size_bytes / 1024**3, 1),
         "short_desc": m.short_desc}
        for m in list_models()
    ]}


@app.get("/api/status")
def api_status():
    s = MANAGER.status()
    s["history_turns"] = {k: len(v) for k, v in HISTORY.items()}
    return s


# ---------------------------------------------------------------------------
# Load / unload / reset — the UI buttons
# ---------------------------------------------------------------------------
class LoadRequest(BaseModel):
    model_id: str
    device: str = "mp"             # mp | cuda | cpu
    expert_offload: bool = False   # MoE: keep experts in CPU RAM


@app.post("/api/load")
def api_load(req: LoadRequest):
    try:
        lm = MANAGER.get_or_load(req.model_id, req.device, req.expert_offload)
        HISTORY.pop(req.model_id, None)  # fresh context on (re)load
        return {"ok": True, "model_id": lm.spec.id, "device": lm.device,
                "display_name": lm.spec.display_name}
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        raise HTTPException(500, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.post("/api/unload")
def api_unload():
    ok = MANAGER.unload()
    HISTORY.clear()
    return {"ok": ok}


@app.post("/api/reset")
def api_reset():
    """Reset the conversation context (drops history)."""
    if MANAGER.current is not None:
        mid = MANAGER.current.spec.id
        HISTORY.pop(mid, None)
        MANAGER.reset_context()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat (SSE streaming)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    model_id: str | None = None   # if set and differs from loaded -> swap
    device: str = "mp"
    expert_offload: bool = False
    max_new_tokens: int = 8192


@app.post("/api/chat")
def chat(req: ChatRequest):
    import torch  # noqa: PLC0415

    # Decide which model to use: the requested one, or the currently loaded one.
    model_id = req.model_id
    if model_id is None:
        if MANAGER.current is None:
            raise HTTPException(400, "no model loaded; call /api/load first")
        model_id = MANAGER.current.spec.id

    try:
        lm = MANAGER.get_or_load(model_id, req.device, req.expert_offload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"load failed: {type(e).__name__}: {e}")

    tokenizer = lm.tokenizer
    model = lm.model
    info = lm.info
    history = HISTORY.setdefault(lm.spec.id, [])

    messages = list(history)
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

            yield sse({"type": "prompt", "tokens": n_prompt,
                       "model_id": lm.spec.id, "moe": lm.spec.moe,
                       "device": lm.device})
            yield sse({"type": "ctx", "used": n_prompt, "max": info.max_position_embeddings,
                       "kv_bytes": info.bytes_per_token * n_prompt,
                       "kv_bytes_hr": fmt_bytes(info.bytes_per_token * n_prompt),
                       "kv_bytes_per_token": info.bytes_per_token})
            # Snapshot VRAM at the start so the UI shows what the model occupies.
            yield sse({"type": "vram", "gpus": MANAGER.vram_summary()})

            streamer = make_streamer(tokenizer)
            stats = GenerationStats(prompt_tokens=n_prompt)

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
                    MANAGER.touch()
                    accumulated += piece
                    stats.on_token()
                    thinking, answer = split_thinking(accumulated, saw_think_end)
                    if "</think>" in piece:
                        saw_think_end = True
                    if thinking != last_thinking:
                        yield sse({"type": "token", "phase": "thinking",
                                   "text": thinking[len(last_thinking):]})
                        last_thinking = thinking
                    if answer != last_answer:
                        yield sse({"type": "token", "phase": "answer",
                                   "text": answer[len(last_answer):]})
                        last_answer = answer
                    seq_len = n_prompt + stats.n_new_tokens
                    yield sse({"type": "stats_live",
                               "tps": round(stats.tokens_per_s, 2),
                               "n": stats.n_new_tokens})
                    yield sse({"type": "ctx", "used": seq_len,
                               "max": info.max_position_embeddings,
                               "kv_bytes": info.bytes_per_token * seq_len,
                               "kv_bytes_hr": fmt_bytes(info.bytes_per_token * seq_len),
                               "kv_bytes_per_token": info.bytes_per_token})
            finally:
                thread.join()
                MANAGER.touch()

            elapsed = time.perf_counter() - t_start
            # Persist the turn into history.
            history.append({"role": "user", "content": req.message})
            history.append({"role": "assistant", "content": last_answer})
            yield sse({"type": "stats", "tps": round(stats.tokens_per_s, 2),
                       "n": stats.n_new_tokens, "elapsed": round(elapsed, 2),
                       "prompt_tokens": n_prompt})
            yield sse({"type": "vram", "gpus": MANAGER.vram_summary()})
            yield sse({"type": "done"})
        except Exception as e:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415
    port = int(os.environ.get("PORT", "8000"))
    # Pass the app object (not the "module:app" string) to avoid a re-import
    # that would re-run module-level setup a second time.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info",
                timeout_keep_alive=300)
