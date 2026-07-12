#!/bin/bash
# =============================================================================
# Run the qwen3_l benchmark / inference inside the GPU container.
#
# The HF cache directory holding the model snapshot is bind-mounted so the
# container sees it at the same host path (CKPTDIR). Pass the devices / args
# you want forwarded to bench.py after `--`.
#
# Usage:
#   docker/run.sh /mnt/nvme/huggingface                      # default: cuda+cpu bench
#   docker/run.sh /mnt/nvme/huggingface -- --devices cuda    # GPU only
#   docker/run.sh /mnt/nvme/huggingface -- python3 -m src.inference --device cuda
# =============================================================================
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
shift || true
EXTRA=("$@")

IMG_NAME="qwen3_l:latest"
CONTAINER_NAME="qwen3_l_run"

# Sanity: warn (but still try) if the checkpoint isn't where we expect.
if [ ! -d "$CKPTDIR/models--Qwen--Qwen3-4B-Thinking-2507-FP8" ]; then
    echo "⚠️  WARNING: model snapshot not found under $CKPTDIR"
    echo "    Run downloader/download_qwen3_4B_FP8_THINKING.sh \"$CKPTDIR\" first."
fi

# If the caller passed a leading `--`, drop it (we pass args straight through).
if [ "${EXTRA[0]:-}" = "--" ]; then
    EXTRA=("${EXTRA[@]:1}")
fi

# Default command when nothing extra is supplied: run the full GPU+CPU benchmark.
if [ ${#EXTRA[@]} -eq 0 ]; then
    CMD=(python3 -m bench.bench --ckptdir "$CKPTDIR" --devices cuda cpu)
else
    # Substitute the placeholder ckptdir so callers don't repeat the path.
    CMD=("${EXTRA[@]//\{\{CKPTDIR\}\}/$CKPTDIR}")
    # If the user invoked a module that accepts --ckptdir, inject it for convenience.
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --mount type=bind,src="$CKPTDIR",target="$CKPTDIR" \
    "$IMG_NAME" \
    "${CMD[@]}"
