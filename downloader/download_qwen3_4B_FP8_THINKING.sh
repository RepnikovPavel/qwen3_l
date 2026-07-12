#!/bin/bash
# =============================================================================
# Downloader for Qwen3-4B-Thinking-2507-FP8
#
# Downloads the model into an HF-cache-style layout so it can be loaded with
# `local_files_only=True` (no network access at inference time).
#
# Usage:
#   ./download_qwen3_4B_FP8_THINKING.sh [CKPTDIR]
#       CKPTDIR  destination root (default: /mnt/nvme/huggingface)
#
# The model lands at:
#   $CKPTDIR/models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/main/
# =============================================================================
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-4B-Thinking-2507-FP8"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-4B-Thinking-2507-FP8/snapshots/main"

echo "STATUS: Creating HF cache structure at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: Failed to create directory"; exit 1; }
cd "$SNAPSHOT_DIR" || { echo "ERROR: Failed to change directory"; exit 1; }

echo "STATUS: Downloading configuration and tokenizer files..."
# Resilient per-file download: skip files already present (non-empty), retry
# forever on transient failures, and treat only the essential files as fatal.
ESSENTIAL_FILES="config.json generation_config.json tokenizer.json"
for file in .gitattributes LICENSE README.md config.json generation_config.json \
            merges.txt tokenizer.json tokenizer_config.json vocab.json; do
    if [[ -s "$file" ]]; then
        echo "SKIP (exists): $file"
        continue
    fi
    if ! wget -q --show-progress --continue --tries=20 --read-timeout=30 --waitretry=3 \
            "https://huggingface.co/$MODEL_ID/resolve/main/$file"; then
        if echo "$ESSENTIAL_FILES" | grep -qw "$file"; then
            echo "ERROR: Failed to download essential file $file"
            exit 1
        else
            echo "WARN: Failed to download non-essential file $file (continuing)"
        fi
    fi
done

echo "STATUS: Downloading model weights (~5.2GB)..."
WEIGHT_FILE="model.safetensors"
if ! wget --show-progress --continue \
        --tries=0 --read-timeout=30 --waitretry=5 \
        "https://huggingface.co/$MODEL_ID/resolve/main/$WEIGHT_FILE"; then
    echo "ERROR: Failed to download $WEIGHT_FILE"
    exit 1
fi

echo "STATUS: Verifying file structure..."
WEIGHT_COUNT=$(ls *.safetensors 2>/dev/null | wc -l)
CONFIG_COUNT=$(ls config.json generation_config.json 2>/dev/null | wc -l)
TOKENIZER_CHECK=$([[ -f tokenizer.json ]] && echo "1" || echo "0")

if [ "$WEIGHT_COUNT" -eq 1 ] && [ "$CONFIG_COUNT" -eq 2 ] && [ "$TOKENIZER_CHECK" -eq 1 ]; then
    echo "VERIFICATION: PASSED"
    echo "WEIGHTS: $WEIGHT_COUNT/1"
    echo "CONFIGS: $CONFIG_COUNT/2"
    echo "TOKENIZER: OK"
    du -sh *.safetensors 2>/dev/null
    echo "STATUS: SUCCESS - Model Qwen3-4B-Thinking-2507-FP8 is ready"
    exit 0
else
    echo "VERIFICATION: FAILED"
    echo "WEIGHTS: $WEIGHT_COUNT/1"
    echo "CONFIGS: $CONFIG_COUNT/2"
    echo "TOKENIZER: $TOKENIZER_CHECK"
    echo "STATUS: ERROR - Incomplete model structure"
    exit 1
fi
