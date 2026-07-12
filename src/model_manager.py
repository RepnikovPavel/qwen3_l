"""
Model manager: lazy load/unload, device selection, MoE expert offload.

This is the single place that owns loaded models in the server process. It
provides:

  - load(model_id, device, ...)   — load a model onto a device (cuda/mp/cpu),
                                     with an optional CPU-expert offload for MoE.
  - unload()                       — free the current model + reclaim VRAM.
  - get_or_load(...)               — lazy: load on first use; auto-unload after
                                     IDLE_TIMEOUT seconds of inactivity.
  - vram_summary()                 — per-GPU used/free MiB for the UI.
  - reset_context()                — drop the conversation history (cheap).

The manager is process-global (one model loaded at a time) because the GPUs are
small (2x16 GB) and swapping two models in VRAM simultaneously is not viable.

MoE expert offload (Qwen3-30B-A3B):
  In Qwen3-MoE every layer's experts live in two big stacked tensors
  (gate_up_proj, down_proj) of shape (num_experts, ...) inside one module — see
  docs/Qwen3MoE.md. So we offload at the LAYER granularity: each MoE layer's
  `mlp.experts` can be kept on CPU RAM and pulled to GPU on demand. We expose
  that as device="mp_cpu_offload": attention + router stay on GPU, expert
  weight tensors are moved to CPU and copied back per forward (slower but fits
  31 GB into 2x16 GB).
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.layer_streamer import LayerStreamer  # noqa: E402
from src.models import ModelSpec, get_model  # noqa: E402

# Auto-unload after this many seconds with no requests. Keeps the GPUs free when
# nobody is talking to the demo. Override via DEMO_IDLE_TIMEOUT env.
IDLE_TIMEOUT = float(__import__("os").environ.get("DEMO_IDLE_TIMEOUT", "600"))


@dataclass
class LoadedModel:
    spec: ModelSpec
    tokenizer: object
    model: object
    device: str          # "cuda" / "mp" / "cpu" / "mp_cpu_offload"
    info: object         # ModelInfo (KV budget etc.)
    placement: dict
    loaded_at: float
    last_used: float


class ModelManager:
    """Process-global owner of the currently loaded model."""

    # Double-buffered layer streaming: number of layers to prefetch ahead on
    # each GPU. Larger chunk = more VRAM used but better overlap of copy/compute.
    # Override via env (DEMO_STREAM_CHUNK).
    _STREAM_CHUNK = int(__import__("os").environ.get("DEMO_STREAM_CHUNK", "2"))

    def __init__(self, ckptdir: str):
        self.ckptdir = ckptdir
        self._loaded: Optional[LoadedModel] = None
        self._streamer: Optional[LayerStreamer] = None
        self._lock = threading.RLock()  # serializes load/unload/generate
        self._reaper = threading.Thread(target=self._idle_watch, daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------ public

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._loaded

    def status(self) -> dict:
        with self._lock:
            if self._loaded is None:
                return {"loaded": False, "model_id": None, "device": None,
                        "vram": self.vram_summary()}
            lm = self._loaded
            return {
                "loaded": True,
                "model_id": lm.spec.id,
                "display_name": lm.spec.display_name,
                "device": lm.device,
                "moe": lm.spec.moe,
                "loaded_at": lm.loaded_at,
                "last_used_ago_s": round(time.time() - lm.last_used, 1),
                "idle_timeout_s": IDLE_TIMEOUT,
                "vram": self.vram_summary(),
            }

    def get_or_load(self, model_id: str, device: str = "mp",
                    expert_offload: bool = False) -> LoadedModel:
        """Lazy load: return the current model if it matches, else load.

        If a *different* model is currently loaded, it is unloaded first
        (there's only room for one). Resets the conversation context on swap.
        """
        with self._lock:
            if (self._loaded is not None
                    and self._loaded.spec.id == model_id
                    and self._loaded.device == device):
                self._loaded.last_used = time.time()
                return self._loaded
            if self._loaded is not None:
                self._do_unload()
            return self._do_load(model_id, device, expert_offload)

    def unload(self) -> bool:
        with self._lock:
            if self._loaded is None:
                return False
            self._do_unload()
            return True

    def reset_context(self):
        """Reset is handled by dropping conversation history at the API layer
        (cheap). Nothing to do to the model itself, but this keeps the GPU
        generation state clean by clearing any cached state pointers."""
        with self._lock:
            if self._loaded is None:
                return
            # Past-Key-Value cache lives inside model.generate per call, so there
            # is no persistent state to drop here. The API layer drops history.
            self._loaded.last_used = time.time()

    def touch(self):
        """Mark activity (used by the idle watcher to keep the model hot)."""
        with self._lock:
            if self._loaded is not None:
                self._loaded.last_used = time.time()

    def vram_summary(self) -> list[dict]:
        """Per-GPU memory snapshot for the UI."""
        try:
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                return []
            out = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                out.append({
                    "gpu": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mib": round(total / 1024**2, 1),
                    "used_mib": round(used / 1024**2, 1),
                    "free_mib": round(free / 1024**2, 1),
                })
            return out
        except Exception as e:  # noqa: BLE001
            return [{"error": f"{type(e).__name__}: {e}"}]

    # ------------------------------------------------------------ load/unload

    def _do_load(self, model_id: str, device: str, expert_offload: bool) -> LoadedModel:
        from src.inject_kernel import inject_fp8_kernel  # noqa: PLC0415
        from src.model_info import ModelInfo  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        spec = get_model(model_id)
        # FP8 kernel must be registered before the model touches FP8 layers.
        inject_fp8_kernel()

        path = spec.path(self.ckptdir)
        print(f"[manager] loading {spec.id} ({spec.family}) device={device} "
              f"expert_offload={expert_offload} from {path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

        attn = "eager" if device == "cpu" else "sdpa"
        placement: dict = {}

        if device == "mp" and torch.cuda.device_count() >= 2:
            if expert_offload and spec.moe:
                # FP8 + device_map with CPU is rejected by transformers, so we
                # load the whole model to CPU and stream decoder layers onto a
                # single GPU a chunk at a time (double-buffered, like the user's
                # iter_forward_gpu_buff). The shared rotary_emb + KV cache pin
                # us to ONE gpu; both cards are exercised via parallel instances
                # in the bench instead.
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype="auto", device_map="cpu",
                    local_files_only=True, attn_implementation=attn,
                ).eval()
                # For inference we don't need the MoE load-balancing aux loss, and
                # collecting router_logits through the streaming forward is fiddly
                # (layers return a bare hidden Tensor when use_cache=False). Turn
                # the flag off so Qwen3MoeForCausalLM.forward doesn't read
                # outputs.router_logits.
                if hasattr(model.config, "output_router_logits"):
                    model.config.output_router_logits = False
                gpu_id = int(__import__("os").environ.get("DEMO_GPU_ID", "0"))
                self._streamer = LayerStreamer(
                    model.model, gpu=gpu_id, chunk=self._STREAM_CHUNK)
                self._streamer.install()
                # lm_head lives on the ForCausalLM wrapper, not model.model —
                # move it to the streamer's GPU so logits match the last layer.
                if hasattr(model, "lm_head") and model.lm_head is not None:
                    try:
                        model.lm_head.to(f"cuda:{gpu_id}")
                    except Exception:
                        pass
                placement = {"mode": "layer_stream", "gpu": gpu_id,
                             "chunk": self._STREAM_CHUNK,
                             "n_layers": len(model.model.layers)}
                device = "mp_cpu_offload"
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype="auto", device_map="auto",
                    local_files_only=True, attn_implementation=attn,
                ).eval()
                placement = dict(getattr(model, "hf_device_map", {}))
        elif device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", device_map="cpu",
                local_files_only=True, attn_implementation="eager",
            ).eval()
            from src.dequant import dequantize_model  # noqa: PLC0415
            dequantize_model(model)
            model = model.to(torch.bfloat16)
            placement = {"cpu": "all"}
        else:
            # single GPU (cuda:<id>) — for MoE we still need offload to fit.
            gpu_id = int(__import__("os").environ.get("DEMO_GPU_ID", "0"))
            if expert_offload and spec.moe:
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype="auto", device_map="cpu",
                    local_files_only=True, attn_implementation=attn,
                ).eval()
                self._streamer = LayerStreamer(
                    model.model, gpu=gpu_id, chunk=self._STREAM_CHUNK)
                self._streamer.install()
                placement = {"mode": "layer_stream", "gpu": gpu_id,
                             "chunk": self._STREAM_CHUNK}
                device = "mp_cpu_offload"
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype="auto", device_map="cpu",
                    local_files_only=True, attn_implementation=attn,
                ).eval().to(f"cuda:{gpu_id}")
                placement = {f"cuda:{gpu_id}": "all"}

        info = ModelInfo.from_model(model)
        now = time.time()
        self._loaded = LoadedModel(
            spec=spec, tokenizer=tokenizer, model=model,
            device=device, info=info, placement=placement,
            loaded_at=now, last_used=now,
        )
        self._log_vram("after load")
        print(f"[manager] ready: {spec.id} device={device} "
              f"KV/tok={info.bytes_per_token}B max={info.max_position_embeddings}", flush=True)
        return self._loaded

    def _do_unload(self):
        lm = self._loaded
        print(f"[manager] unloading {lm.spec.id} (device={lm.device})", flush=True)
        # Detach the layer streamer hooks first (if any).
        if self._streamer is not None:
            try:
                self._streamer.remove()
            except Exception:
                pass
            self._streamer = None
        try:
            # Move model to CPU first so VRAM is actually released on .del.
            import torch  # noqa: PLC0415
            lm.model.to("cpu")
        except Exception:
            pass
        self._loaded = None
        del lm
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        self._log_vram("after unload")

    # ------------------------------------------------------------- MoE offload
    # (Replaced by src/layer_streamer.py — double-buffered layer paging that
    # keeps both GPUs busy. The old per-expert hooks + manual device bridge are
    # gone; LayerStreamer installs forward pre-hooks on every decoder layer.)

    # --------------------------------------------------------------- internals

    def _idle_watch(self):
        """Background thread: unload when idle longer than IDLE_TIMEOUT."""
        while True:
            time.sleep(30)
            with self._lock:
                if self._loaded is None:
                    continue
                idle = time.time() - self._loaded.last_used
                if idle >= IDLE_TIMEOUT:
                    print(f"[manager] idle for {idle:.0f}s >= {IDLE_TIMEOUT}s, "
                          f"auto-unloading {self._loaded.spec.id}", flush=True)
                    self._do_unload()

    def _log_vram(self, tag: str):
        v = self.vram_summary()
        if v and "error" not in v[0]:
            parts = [f"GPU{g['gpu']}: {g['used_mib']}/{g['total_mib']} MiB" for g in v]
            print(f"[vram {tag}] " + " | ".join(parts), flush=True)
