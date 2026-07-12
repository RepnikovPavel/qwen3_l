#!/bin/bash
# =============================================================================
# Downloader for Qwen3-30B-A3B-Thinking-2507-FP8 (sharded, MoE).
#
# This model is split across 4 safetensors shards (~31 GB total). The list of
# weight shards is read from model.safetensors.index.json so future reshardings
# are picked up automatically.
#
# Usage:
#   ./download_qwen3_30B_A3B_FP8_THINKING.sh [CKPTDIR]
#       CKPTDIR  destination root (default: /mnt/nvme/huggingface)
#
# The model lands at:
#   $CKPTDIR/models--Qwen--Qwen3-30B-A3B-Thinking-2507-FP8/snapshots/main/
# =============================================================================
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-30B-A3B-Thinking-2507-FP8/snapshots/main"

echo "STATUS: Creating HF cache structure at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: Failed to create directory"; exit 1; }
cd "$SNAPSHOT_DIR" || { echo "ERROR: Failed to change directory"; exit 1; }

echo "STATUS: Downloading configuration and tokenizer files..."
ESSENTIAL_FILES="config.json generation_config.json tokenizer.json"
for file in .gitattributes LICENSE README.md config.json generation_config.json \
            merges.txt tokenizer.json tokenizer_config.json vocab.json \
            model.safetensors.index.json; do
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

# Read the shard list from the index json (robust to resharding).
if [[ ! -s model.safetensors.index.json ]]; then
    echo "ERROR: model.safetensors.index.json missing — cannot determine shards"
    exit 1
fi

mapfile -t SHARDS < <(python3 -c "
import json
idx = json.load(open('model.safetensors.index.json'))
files = sorted(set(idx['weight_map'].values()))
for f in files:
    print(f)
")

echo "STATUS: Downloading ${#SHARDS[@]} weight shards (~31 GB total)..."
for shard in "${SHARDS[@]}"; do
    if [[ -s "$shard" ]]; then
        echo "SKIP (exists): $shard"
        continue
    fi
    echo "STATUS: downloading $shard ..."
    if ! wget --show-progress --continue \
            --tries=0 --read-timeout=30 --waitretry=5 \
            "https://huggingface.co/$MODEL_ID/resolve/main/$shard"; then
        echo "ERROR: Failed to download $shard"
        exit 1
    fi
done

echo "STATUS: Verifying file structure..."
WEIGHT_COUNT=$(ls model-*.safetensors 2>/dev/null | wc -l)
CONFIG_OK=$([[ -s config.json ]] && echo "1" || echo "0")
TOKENIZER_OK=$([[ -s tokenizer.json ]] && echo "1" || echo "0")
INDEX_OK=$([[ -s model.safetensors.index.json ]] && echo "1" || echo "0")

echo "WEIGHTS: $WEIGHT_COUNT shards"
echo "CONFIG: $CONFIG_OK"
echo "TOKENIZER: $TOKENIZER_OK"
echo "INDEX: $INDEX_OK"

EXPECTED=${#SHARDS[@]}
if [ "$WEIGHT_COUNT" -ge "$EXPECTED" ] && [ "$CONFIG_OK" = "1" ] \
   && [ "$TOKENIZER_OK" = "1" ] && [ "$INDEX_OK" = "1" ]; then
    echo "VERIFICATION: PASSED"
    du -sh model-*.safetensors 2>/dev/null
    echo "STATUS: SUCCESS - Model Qwen3-30B-A3B-Thinking-2507-FP8 is ready"
    exit 0
else
    echo "VERIFICATION: FAILED"
    echo "STATUS: ERROR - Incomplete model structure"
    exit 1
fi
