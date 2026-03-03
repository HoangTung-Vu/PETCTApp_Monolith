#!/bin/bash
# push_images.sh — Build and push AI_engine Docker images to DockerHub
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

# Local image names (used during build)
NNUNET_IMAGE="${ENGINE_NNUNET_IMAGE:-engine-nnunet}"
NNUNET_OLD_IMAGE="${ENGINE_NNUNET_OLD_IMAGE:-engine-nnunet-old-ver}"
AUTOPET_IMAGE="${ENGINE_AUTOPET_IMAGE:-engine-autopet}"
TOTALSEG_IMAGE="${ENGINE_TOTALSEG_IMAGE:-engine-totalseg}"

# Remote image names on DockerHub
REMOTE_NNUNET="${DOCKERHUB_USER}/${NNUNET_IMAGE}"
REMOTE_NNUNET_OLD="${DOCKERHUB_USER}/${NNUNET_OLD_IMAGE}"
REMOTE_AUTOPET="${DOCKERHUB_USER}/${AUTOPET_IMAGE}"
REMOTE_TOTALSEG="${DOCKERHUB_USER}/${TOTALSEG_IMAGE}"

TAG="${IMAGE_TAG:-latest}"

# --- Helper Functions ---
build_image() {
    local image_name="$1"
    local context_dir="$2"
    echo "======================================="
    echo "  Building: $image_name"
    echo "  Context:  $context_dir"
    echo "======================================="
    docker build -t "$image_name" "$context_dir"
}

tag_and_push() {
    local local_name="$1"
    local remote_name="$2"
    local tag="$3"

    echo "---------------------------------------"
    echo "  Tagging:  ${local_name} -> ${remote_name}:${tag}"
    echo "---------------------------------------"
    docker tag "$local_name" "${remote_name}:${tag}"

    echo "  Pushing:  ${remote_name}:${tag}"
    docker push "${remote_name}:${tag}"
    echo "  ✓ Pushed ${remote_name}:${tag}"
}

# --- Main Execution ---
echo "============================================"
echo "  PET/CT AI Engines — Build & Push to Hub"
echo "============================================"

# Ensure logged in
echo -e "\n-- Checking DockerHub login --"
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "You are not logged in to DockerHub. Logging in..."
    docker login
fi

# Step 1: Build images
echo -e "\n-- Step 1: Building Docker images --"
build_image "$NNUNET_IMAGE"     "$SCRIPT_DIR/AI_engines/engine_nnunet"
build_image "$NNUNET_OLD_IMAGE" "$SCRIPT_DIR/AI_engines/engine_nnunet_old_ver"
build_image "$AUTOPET_IMAGE"    "$SCRIPT_DIR/AI_engines/engine_autopet"
build_image "$TOTALSEG_IMAGE"   "$SCRIPT_DIR/AI_engines/engine_totalseg"

# Step 2: Tag & Push
echo -e "\n-- Step 2: Tagging & Pushing to DockerHub --"
tag_and_push "$NNUNET_IMAGE"     "$REMOTE_NNUNET"     "$TAG"
tag_and_push "$NNUNET_OLD_IMAGE" "$REMOTE_NNUNET_OLD" "$TAG"
tag_and_push "$AUTOPET_IMAGE"    "$REMOTE_AUTOPET"    "$TAG"
tag_and_push "$TOTALSEG_IMAGE"   "$REMOTE_TOTALSEG"   "$TAG"

echo ""
echo "============================================"
echo "  ✓ All images pushed successfully!"
echo "============================================"
echo ""
echo "  Images on DockerHub:"
echo "    • ${REMOTE_NNUNET}:${TAG}"
echo "    • ${REMOTE_NNUNET_OLD}:${TAG}"
echo "    • ${REMOTE_AUTOPET}:${TAG}"
echo "    • ${REMOTE_TOTALSEG}:${TAG}"
echo ""
