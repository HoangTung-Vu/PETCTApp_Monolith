#!/bin/bash
# push_totalseg.sh — Build the TotalSegmentator engine image, push it to Docker Hub,
# then remove the local copy (docker rmi).
#
# Weights for TotalSegmentator task "total" (open, Apache-2.0) are baked in at build time.
# On the GPU server:
#   docker run -d --name totalseg-engine --gpus all --ipc=host -p 8105:8105 \
#       hoangtungmvu/totalseg-engine:2.18.0
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKERHUB_USER="${DOCKERHUB_USER:-hoangtungmvu}"
IMAGE="${DOCKERHUB_USER}/${ENGINE_TOTALSEG_IMAGE:-totalseg-engine}"
TAG="${IMAGE_TAG:-2.18.0}"   # = pinned TotalSegmentator version

echo "============================================"
echo "  TotalSegmentator Engine — Build & Push"
echo "============================================"

echo -e "\n-- Checking Docker Hub login --"
if ! docker info 2>/dev/null | grep -q "Username"; then
    docker login
fi

echo -e "\n-- Step 1: Build ${IMAGE}:${TAG} --"
docker build -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" "$SCRIPT_DIR/AI_engines/engine_totalseg"

echo -e "\n-- Step 2: Push --"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

echo -e "\n-- Step 3: Remove local image --"
docker rmi "${IMAGE}:${TAG}" "${IMAGE}:latest"

echo -e "\nPushed: ${IMAGE}:${TAG} (+ :latest), local copy removed."
