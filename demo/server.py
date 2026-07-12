"""
FastAPI web demo server for Qwen3 (4B dense + 30B MoE), with session
persistence and interruptible generation.

Key behaviors requested:
  - Chat sessions persist in SQLite; refreshing the browser tab does NOT lose
    history and does NOT abort a running generation (the server keeps
    generating into the session even if the SSE client disconnects).
  - "Reset context" clears the live KV cache + the in-memory working history
    (GPU state), but leaves saved sessions untouched.
  - A "stop" button interrupts the active generation for a session.
  - Models are lazy-loaded on first request and auto-unloaded after idle.

Endpoints:
  GET  /                          UI
  GET  /api/models                list supported models
  GET  /api/status                loaded model + per-GPU VRAM
  POST /api/load                  (model_id, device, expert_offload)
  POST /api/unload
  POST /api/reset                 clear live KV cache / working history
  GET  /api/sessions              list saved sessions
  POST /api/sessions              create a session -> {id}
  GET  /api/sessions/{id}         session + messages
  PATCH /api/sessions/{id}        rename
  DELETE /api/sessions/{id}
  POST /api/chat                  SSE stream of one assistant turn
  POST /api/stop/{session_id}     stop the active generation for a session
  GET  /api/stream/{session_id}   re-attach to an in-flight generation SSE
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException, Request
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
from demo.sessions import SessionStore  # noqa: E402

app = FastAPI(title="Qwen3 demo (4B + 30B-MoE)")

CKPTDIR = os.environ.get("CKPTDIR", "/mnt/nvme/huggingface")
MANAGER = ModelManager(CKPTDIR)

# Sessions live on disk — survive page refresh / reconnect.
DB_PATH = os.environ.get("DEMO_DB", str(Path.home() / "qwen3_demo_sessions.sqlite"))
STORE = SessionStore(DB_PATH)

# In-flight generations: session_id -> Generation. Cleared on completion/stop.
# This is the "GPU working context": the live KV cache is owned by model.generate
# per call, but we keep the partial outputs here so a refreshed tab can re-stream.
GENERATIONS: dict[str, "Generation"] = {}
GEN_LOCK = threading.Lock()


class Generation:
    """One in-flight (or completed) generation for a session.

    The server runs generate() on a background thread, writing SSE frames into
    a thread-safe queue. The /api/chat response streams from the queue; if the
    client disconnects, the /api/stream/<sid> endpoint can re-attach to the
    same queue. The queue is bounded-ish (a list + Condition).
    """
    def __init__(self, session_id: str, prompt: str):
        self.session_id = session_id
        self.prompt = prompt
        self.frames: list[str] = []        # all frames produced so far
        self.cv = threading.Condition()    # signal on new frame / completion
        self.done = False
        self.error: str | None = None
        self.stop_flag = threading.Event()
        self.stats = GenerationStats()
        self.started_at = time.time()

    def emit(self, frame: str):
        with self.cv:
            self.frames.append(frame)
            self.cv.notify_all()

    def finish(self, error: str | None = None):
        with self.cv:
            self.done = True
            self.error = error
            self.cv.notify_all()

    def stop(self):
        self.stop_flag.set()

    def iter_frames(self, from_index: int = 0):
        """Generator yielding frames from `from_index`, live until done."""
        i = from_index
        while True:
            with self.cv:
                while i >= len(self.frames) and not self.done:
                    self.cv.wait(timeout=1.0)
                new = self.frames[i:]
                done = self.done
                err = self.error
            for f in new:
                yield f
            i += len(new)
            if i >= len(self.frames) and done:
                if err and i == len(self.frames):
                    yield f"data: {json.dumps({'type': 'error', 'message': err})}\n\n"
                return


# ===========================================================================
# Static / metadata
# ===========================================================================
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
    with GEN_LOCK:
        s["active_generations"] = list(GENERATIONS.keys())
    s["ram"] = MANAGER.ram_summary()
    return s


@app.get("/api/memory")
def api_memory():
    """Dedicated memory snapshot for the UI's 'how much GPU/CPU we consume' panel."""
    # Snapshot current ONCE to avoid a TOCTOU race with the idle-watcher that
    # nulls MANAGER.current between the `is not None` check and the `.spec` read.
    cur = MANAGER.current
    return {
        "vram": MANAGER.vram_summary(),
        "ram": MANAGER.ram_summary(),
        "loaded": cur is not None,
        "model_id": cur.spec.id if cur is not None else None,
    }


# ===========================================================================
# Load / unload / reset
# ===========================================================================
class LoadRequest(BaseModel):
    model_id: str
    device: str = "mp"
    expert_offload: bool = False


@app.post("/api/load")
def api_load(req: LoadRequest):
    try:
        lm = MANAGER.get_or_load(req.model_id, req.device, req.expert_offload)
        return {"ok": True, "model_id": lm.spec.id, "device": lm.device,
                "display_name": lm.spec.display_name}
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        raise HTTPException(500, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.post("/api/unload")
def api_unload():
    ok = MANAGER.unload()
    return {"ok": ok}


@app.post("/api/reset")
def api_reset():
    """Reset the live GPU working context (KV cache + in-memory history).

    Saved SQLite sessions are NOT touched — only the live state.
    """
    MANAGER.reset_context()
    return {"ok": True}


# ===========================================================================
# Sessions
# ===========================================================================
class CreateSessionRequest(BaseModel):
    model_id: str | None = None
    title: str | None = None


@app.get("/api/sessions")
def api_list_sessions():
    return {"sessions": STORE.list_sessions()}


@app.post("/api/sessions")
def api_create_session(req: CreateSessionRequest):
    return STORE.create_session(model_id=req.model_id, title=req.title)


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    s = STORE.get_session(session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    # Attach generation status if active.
    with GEN_LOCK:
        gen = GENERATIONS.get(session_id)
    if gen is not None:
        s["generation"] = {"done": gen.done, "error": gen.error,
                           "n_frames": len(gen.frames),
                           "stop_requested": gen.stop_flag.is_set()}
    return s


class RenameRequest(BaseModel):
    title: str


@app.patch("/api/sessions/{session_id}")
def api_rename_session(session_id: str, req: RenameRequest):
    if not STORE.rename_session(session_id, req.title):
        raise HTTPException(404, "session not found")
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    STORE.delete_session(session_id)
    with GEN_LOCK:
        gen = GENERATIONS.get(session_id)
    if gen is not None:
        gen.stop()
    return {"ok": True}


# ===========================================================================
# Chat (background generation + SSE stream)
# ===========================================================================
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model_id: str | None = None
    device: str = "mp"
    expert_offload: bool = False
    max_new_tokens: int = 8192


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _run_generation(gen: Generation, lm, message: str, model_id: str,
                    device: str, expert_offload: bool, max_new_tokens: int):
    """Background worker: load model, run generate() with a streamer, emit SSE.

    The generation keeps going even if the client disconnects — the partial
    result is written into the session store at the end.
    """
    tokenizer = lm.tokenizer
    model = lm.model
    info = lm.info
    # Build message list from the saved session history + new user message.
    sess = STORE.get_session(gen.session_id) or {"messages": []}
    messages = list(sess.get("messages", []))
    messages.append({"role": "user", "content": message})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Inputs to the embedding device.
        if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None):
            input_device = next(iter(model.hf_device_map.values()))
        else:
            embed = getattr(getattr(model, "model", model), "embed_tokens", None)
            if embed is not None:
                try:
                    input_device = next(embed.parameters()).device
                except StopIteration:
                    input_device = model.device
            else:
                input_device = model.device
        inputs = tokenizer([text], return_tensors="pt").to(input_device)
        n_prompt = inputs.input_ids.shape[1]

        gen.emit(_sse({"type": "prompt", "tokens": n_prompt,
                       "model_id": lm.spec.id, "moe": lm.spec.moe,
                       "device": lm.device, "session_id": gen.session_id}))
        gen.emit(_sse({"type": "ctx", "used": n_prompt,
                       "max": info.max_position_embeddings,
                       "kv_bytes": info.bytes_per_token * n_prompt,
                       "kv_bytes_hr": fmt_bytes(info.bytes_per_token * n_prompt),
                       "kv_bytes_per_token": info.bytes_per_token}))
        gen.emit(_sse({"type": "vram", "gpus": MANAGER.vram_summary()}))

        streamer = make_streamer(tokenizer)
        gen.stats.prompt_tokens = n_prompt
        # Custom stopping: abort generate when the stop_flag is set.
        class _StopOnFlag:
            def __init__(self, flag): self.flag = flag
            def __call__(self, ids, scores, **kw): return self.flag.is_set()
            @property
            def input_ids(self): return None
            @property
            def scores(self): return None

        gen_kwargs = dict(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=type("SC", (), {
                "max_length": 10**9,
                "__iter__": lambda self_: iter([_StopOnFlag(gen.stop_flag)]),
                "__len__": lambda self_: 1,
            })(),
        )

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        saw_think_end = False
        last_thinking = ""
        last_answer = ""
        try:
            for piece in streamer:
                if gen.stop_flag.is_set():
                    break
                MANAGER.touch()
                accumulated += piece
                gen.stats.on_token()
                thinking, answer = split_thinking(accumulated, saw_think_end)
                if "</think>" in piece:
                    saw_think_end = True
                if thinking != last_thinking:
                    gen.emit(_sse({"type": "token", "phase": "thinking",
                                   "text": thinking[len(last_thinking):]}))
                    last_thinking = thinking
                if answer != last_answer:
                    gen.emit(_sse({"type": "token", "phase": "answer",
                                   "text": answer[len(last_answer):]}))
                    last_answer = answer
                seq_len = n_prompt + gen.stats.n_new_tokens
                gen.emit(_sse({"type": "stats_live",
                               "tps": round(gen.stats.tokens_per_s, 2),
                               "n": gen.stats.n_new_tokens}))
                gen.emit(_sse({"type": "ctx", "used": seq_len,
                               "max": info.max_position_embeddings,
                               "kv_bytes": info.bytes_per_token * seq_len,
                               "kv_bytes_hr": fmt_bytes(info.bytes_per_token * seq_len),
                               "kv_bytes_per_token": info.bytes_per_token}))
        finally:
            # Signal the generate thread to stop if it's still running.
            gen.stop_flag.set()
            thread.join(timeout=30)

        elapsed = time.time() - gen.started_at
        # Persist the turn (only if anything was produced).
        if message:
            STORE.append_message(gen.session_id, "user", message)
        if last_answer or last_thinking:
            STORE.append_message(gen.session_id, "assistant",
                                 last_answer or last_thinking)
        stopped = gen.stop_flag.is_set()
        gen.emit(_sse({"type": "stats",
                       "tps": round(gen.stats.tokens_per_s, 2),
                       "n": gen.stats.n_new_tokens, "elapsed": round(elapsed, 2),
                       "prompt_tokens": n_prompt, "stopped": stopped}))
        gen.emit(_sse({"type": "vram", "gpus": MANAGER.vram_summary()}))
        gen.emit(_sse({"type": "done", "stopped": stopped}))
        gen.finish()
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        gen.finish(error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """Start a generation and stream SSE frames. If the client disconnects,
    generation continues server-side; re-attach via GET /api/stream/<sid>."""
    # Resolve / create session.
    if req.session_id is None:
        sess = STORE.create_session(model_id=req.model_id)
        req.session_id = sess["id"]
    elif STORE.get_session(req.session_id) is None:
        raise HTTPException(404, f"session {req.session_id} not found")

    # Refuse if a generation is already running for this session.
    with GEN_LOCK:
        existing = GENERATIONS.get(req.session_id)
    if existing is not None and not existing.done:
        raise HTTPException(409, "generation already running for this session; "
                            f"re-attach via /api/stream/{req.session_id}")

    # Make sure a model is loaded.
    model_id = req.model_id
    if model_id is None:
        cur = MANAGER.current  # snapshot to avoid race with idle-watcher
        if cur is None:
            raise HTTPException(400, "no model loaded; call /api/load first")
        model_id = cur.spec.id
    try:
        lm = MANAGER.get_or_load(model_id, req.device, req.expert_offload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"load failed: {type(e).__name__}: {e}")

    gen = Generation(req.session_id, req.message)
    with GEN_LOCK:
        GENERATIONS[req.session_id] = gen

    # Run generation on a daemon thread; the request streams from gen.iter_frames.
    worker = threading.Thread(
        target=_run_generation,
        args=(gen, lm, req.message, model_id, req.device, req.expert_offload,
              req.max_new_tokens),
        daemon=True,
    )
    worker.start()

    # NOTE: the stream must be async to use `await request.is_disconnected()`.
    async def aevent_stream():
        for frame in gen.iter_frames(0):
            if await request.is_disconnected():
                break
            yield frame
        if gen.done:
            with GEN_LOCK:
                GENERATIONS.pop(req.session_id, None)

    headers = {"X-Session-Id": req.session_id,
               "Cache-Control": "no-cache",
               "X-Accel-Buffering": "no"}
    return StreamingResponse(aevent_stream(), media_type="text/event-stream",
                             headers=headers)


@app.post("/api/stop/{session_id}")
def api_stop(session_id: str):
    """Stop the active generation for a session."""
    with GEN_LOCK:
        gen = GENERATIONS.get(session_id)
    if gen is None:
        return {"ok": False, "reason": "no active generation"}
    gen.stop()
    return {"ok": True}


@app.get("/api/stream/{session_id}")
def api_stream(session_id: str, request: Request):
    """Re-attach to an in-flight generation's SSE stream (after page refresh)."""
    with GEN_LOCK:
        gen = GENERATIONS.get(session_id)
    if gen is None:
        raise HTTPException(404, "no active generation for this session")

    async def aevent_stream():
        # Re-play all frames from the start, then continue live.
        for frame in gen.iter_frames(0):
            if await request.is_disconnected():
                break
            yield frame
        if gen.done:
            with GEN_LOCK:
                GENERATIONS.pop(session_id, None)

    return StreamingResponse(aevent_stream(), media_type="text/event-stream",
                             headers={"X-Session-Id": session_id,
                                      "Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info",
                timeout_keep_alive=300)
