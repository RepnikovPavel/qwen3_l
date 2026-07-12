# =============================================================================
# Inference image for Qwen3-4B-Thinking-2507-FP8.
#
# Built on the TensorRT-LLM release image, which already ships:
#   - Python 3.12
#   - PyTorch (CUDA build)
#   - transformers 5.5.x (with hub_kernels._KERNEL_MODULE_MAPPING support)
#   - triton (needed to JIT-compile the FP8 Triton kernels)
# So no extra pip installs are required to run the benchmark — we only copy in
# the source tree. The model checkpoint and kernel folder are bind-mounted at
# runtime so the image stays small and swappable.
# =============================================================================
FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20

WORKDIR /app

# Copy the repo (src/, bench/, kernels/). The model weights and the HF cache
# are bind-mounted at runtime via run.sh to keep the image weight low.
COPY . /app

# Env defaults — overridable at run time. CKPTDIR is the HF cache root that
# contains models--Qwen--Qwen3-4B-Thinking-2507-FP8.
ENV QWEN3_FP8_KERNEL_DIR=/app/kernels/local_kernels_qwen3_8B_FP8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

CMD ["python3", "-m", "bench.bench", "--devices", "cuda", "cpu"]
