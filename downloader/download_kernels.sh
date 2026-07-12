#!/bin/bash
# =============================================================================
# (Optional) Fetch the finegrained-fp8 Triton kernel from the kernels-community
# HF repo into a local folder. This is an ALTERNATIVE to shipping the checked-in
# copy under kernels/local_kernels_qwen3_8B_FP8/ — use it if you want to refresh
# the kernel sources from upstream instead of using the vendored files.
#
# Usage:
#   ./download_kernels.sh [DEST_DIR]
#       DEST_DIR  target directory
#                 (default: ./kernels/local_kernels_qwen3_8B_FP8, relative to repo root)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR=${1:-"$REPO_ROOT/kernels/local_kernels_qwen3_8B_FP8"}

# Create an isolated folder plus a subfolder for the kernel's internal imports.
mkdir -p "$DEST_DIR/finegrained_fp8"

# Base URL of the kernel repository.
BASE_URL="https://huggingface.co/kernels-community/finegrained-fp8/resolve/main/build/torch-cuda"

# Download the root kernel files.
wget ${BASE_URL}/__init__.py     -O "$DEST_DIR/__init__.py"
wget ${BASE_URL}/_ops.py         -O "$DEST_DIR/_ops.py"
wget ${BASE_URL}/act_quant.py    -O "$DEST_DIR/act_quant.py"
wget ${BASE_URL}/batched.py      -O "$DEST_DIR/batched.py"
wget ${BASE_URL}/grouped.py      -O "$DEST_DIR/grouped.py"
wget ${BASE_URL}/matmul.py       -O "$DEST_DIR/matmul.py"
wget ${BASE_URL}/metadata.json   -O "$DEST_DIR/metadata.json"
wget ${BASE_URL}/utils.py        -O "$DEST_DIR/utils.py"

# Download the file from the nested subfolder.
wget ${BASE_URL}/finegrained_fp8/__init__.py -O "$DEST_DIR/finegrained_fp8/__init__.py"

echo "STATUS: SUCCESS - kernels downloaded to $DEST_DIR"
