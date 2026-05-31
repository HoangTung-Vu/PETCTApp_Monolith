#!/bin/bash
# push_images.sh — Build and push the nnU-Net engine image to DockerHub (dev only).
#
# NOTE: Model weights are baked into the Docker image via `COPY weights/` in the
# Dockerfile. Do NOT push to a public registry if your weights are proprietary.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env if available
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# --- Configuration ---
DOCKERHUB_USER="${DOCKERHUB_USER:-}"
if [ -z "$DOCKERHUB_USER" ]; then
    read -rp "Enter your DockerHub username: " DOCKERHUB_USER
fi

NNUNET_IMAGE="${ENGINE_NNUNET_IMAGE:-engine-nnunet}"
REMOTE_NNUNET="${DOCKERHUB_USER}/${NNUNET_IMAGE}"
TAG="${IMAGE_TAG:-latest}"

# --- Main Execution ---
echo "============================================"
echo "  nnU-Net Engine — Build & Push to DockerHub"
echo "============================================"

# Ensure logged in
echo -e "\n-- Checking DockerHub login --"
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "Not logged in to DockerHub. Logging in..."
    docker login
fi

# Step 1: Build
echo -e "\n-- Step 1: Building Docker image --"
echo "  Context: $SCRIPT_DIR/AI_engines/engine_nnunet_old_ver"
docker build -t "$NNUNET_IMAGE" "$SCRIPT_DIR/AI_engines/engine_nnunet_old_ver"

# Step 2: Tag & Push
echo -e "\n-- Step 2: Tagging & Pushing --"
docker tag "$NNUNET_IMAGE" "${REMOTE_NNUNET}:${TAG}"
docker push "${REMOTE_NNUNET}:${TAG}"
echo "  Pushed: ${REMOTE_NNUNET}:${TAG}"
