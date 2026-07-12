#!/bin/bash
# =============================================================================
# Build the qwen3_l inference image.
#
# Usage (from repo root):
#   docker/build.sh [TAG]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG=${1:-"qwen3_l:latest"}

DOCKER_BUILDKIT=1 docker buildx build \
    --pull=false \
    -t "$TAG" \
    -f "$SCRIPT_DIR/Dockerfile" \
    --progress=plain \
    "$REPO_ROOT"

echo "✅ Built $TAG"
docker image ls "$TAG"
