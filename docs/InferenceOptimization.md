# Inference optimization notes (5060 Ti × 2)

## Current stack (qwen3_l)

| Component | Role |
|---|---|
| `inject_kernel.py` | Offline Triton finegrained-fp8 + deep-gemm stub |
| `expert_streamer.py` | MoE expert paging: 2-GPU split + slice-LRU |
| `demo/server.py` | Lazy load, SQLite sessions, VRAM panel |

### Env knobs (30B MoE)

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_EXPERT_SLICE_MODE` | `1` | Per-expert slice LRU (~4.5 MiB) vs whole module (576 MiB) |
| `DEMO_EXPERT_SPLIT_GPU` | `1` | Contiguous-half across 2 GPUs + rotary bridge |
| `DEMO_EXPERT_CACHE_GIB` | `11` | Per-GPU LRU cache budget (GiB) |
| `DEMO_EXPERT_CACHE_SAFETY_GIB` | `1.0` | VRAM headroom before eviction |

### Measured (2026-07-12, 2× RTX 5060 Ti)

| Config | tok/s |
|---|---|
| Layer-LRU, single GPU | 0.24 |
| 2-GPU split (layer-LRU) | 0.24 (PCIe bound on 576 MiB copies) |
| **2-GPU split + slice-LRU** | **2.31** |

Bottleneck after slice-LRU: attention + FP8 matmul on both GPUs (utilization
improves; further gains need fused MoE kernels or a dedicated inference engine).

## Future backends (no extra model downloads required)

### [llama.cpp](https://github.com/ggml-org/llama.cpp)

- GGUF export of existing HF checkpoint → `llama-server` with MoE CPU offload.
- Pros: mature paging, CPU/GPU hybrid, low RSS fragmentation.
- Cons: needs one-time GGUF conversion on server (from weights already in `hf_cache`).

### [Unsloth](https://unsloth.ai/)

- Primarily training/fine-tuning; inference patterns (pinned memory, fused ops)
  already adopted here (DMA copy, slice cache).
- Could accelerate 4B dense path if integrated; 30B MoE still needs custom paging.

### TensorRT-LLM / vLLM

- Base Docker image is TRT-LLM 1.3.0rc20 — natural next step for 4B dense
  and eventually 30B if MoE FP8 paths mature in those engines.