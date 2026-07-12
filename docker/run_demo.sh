#!/bin/bash
# =============================================================================
# Run the Qwen3-4B-Thinking-2507-FP8 web demo in the GPU container.
#
# By default the model runs MODEL-PARALLEL ("flat") across all visible GPUs —
# i.e. layers are split across cards so one model spans them (larger effective
# memory pool). Override with env vars:
#   DEMO_DEVICE=mp           model-parallel across GPUs (default)
#   DEMO_DEVICE=cuda         single GPU (use DEMO_GPU_ID=0/1)
#   DEMO_DEVICE=cpu          CPU (dequantized)
#   DEMO_MP_MAX_MEMORY=0:14GiB,1:14GiB   per-GPU cap for model-parallel
#
# Usage:
#   docker/run_demo.sh /path/to/hf_cache [PORT]
#
# Then from your laptop:
#   ssh -L 8000:localhost:8000 tuna-server
#   open http://localhost:8000
# =============================================================================
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
PORT=${2:-"8000"}
IMG_NAME="qwen3_l:latest"
CONTAINER_NAME="qwen3_l_demo"

if [ ! -d "$CKPTDIR/models--Qwen--Qwen3-4B-Thinking-2507-FP8" ]; then
    echo "⚠️  WARNING: model snapshot not found under $CKPTDIR"
    echo "    Run downloader/download_qwen3_4B_FP8_THINKING.sh \"$CKPTDIR\" first."
fi

echo "→ launching demo on 0.0.0.0:$PORT  (DEMO_DEVICE=${DEMO_DEVICE:-mp})"
echo "→ access via: ssh -L ${PORT}:localhost:${PORT} tuna-server  →  http://localhost:${PORT}"

docker run --rm \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e CKPTDIR="$CKPTDIR" \
    -e DEMO_DEVICE="${DEMO_DEVICE:-mp}" \
    -e DEMO_GPU_ID="${DEMO_GPU_ID:-0}" \
    -e DEMO_MP_MAX_MEMORY="${DEMO_MP_MAX_MEMORY:-}" \
    -e PORT="$PORT" \
    --mount type=bind,src="$CKPTDIR",target="$CKPTDIR" \
    "$IMG_NAME" \
    python3 -m demo.server
