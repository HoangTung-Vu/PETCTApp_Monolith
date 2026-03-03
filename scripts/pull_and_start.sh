#!/bin/bash
# pull_and_start.sh — Pull AI_engine images from DockerHub and start the app
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

TAG="${IMAGE_TAG:-latest}"

# Ports
NNUNET_PORT="${ENGINE_NNUNET_PORT:-8101}"
NNUNET_OLD_PORT="${ENGINE_NNUNET_OLD_PORT:-8104}"
AUTOPET_PORT="${ENGINE_AUTOPET_PORT:-8102}"
TOTALSEG_PORT="${ENGINE_TOTALSEG_PORT:-8103}"

# Remote image names on DockerHub
NNUNET_IMAGE="${DOCKERHUB_USER}/${ENGINE_NNUNET_IMAGE:-engine-nnunet}:${TAG}"
NNUNET_OLD_IMAGE="${DOCKERHUB_USER}/${ENGINE_NNUNET_OLD_IMAGE:-engine-nnunet-old-ver}:${TAG}"
AUTOPET_IMAGE="${DOCKERHUB_USER}/${ENGINE_AUTOPET_IMAGE:-engine-autopet}:${TAG}"
TOTALSEG_IMAGE="${DOCKERHUB_USER}/${ENGINE_TOTALSEG_IMAGE:-engine-totalseg}:${TAG}"

# Container names
NNUNET_CONTAINER="${ENGINE_NNUNET_CONTAINER:-engine-nnunet-container}"
NNUNET_OLD_CONTAINER="${ENGINE_NNUNET_OLD_CONTAINER:-engine-nnunet-old-ver-container}"
AUTOPET_CONTAINER="${ENGINE_AUTOPET_CONTAINER:-engine-autopet-container}"
TOTALSEG_CONTAINER="${ENGINE_TOTALSEG_CONTAINER:-engine-totalseg-container}"

# GPU detection
GPU_FLAG=""
if docker info | grep -i "Runtimes.*nvidia" >/dev/null 2>&1 || nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Enabling GPU passthrough."
    GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU detected. Running CPU mode."
fi

# --- Helper Functions ---
pull_image() {
    local image="$1"
    echo "---------------------------------------"
    echo "  Pulling: $image"
    echo "---------------------------------------"
    docker pull "$image"
    echo "  ✓ Pulled $image"
}

start_container() {
    local container_name="$1"
    local image_name="$2"
    local port="$3"

    # Remove existing container if any
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
    fi

    echo "Starting container: $container_name (port $port)"
    docker run -d \
        --name "$container_name" \
        $GPU_FLAG \
        -p "$port:$port" \
        "$image_name"
}

wait_for_health() {
    local port="$1"
    local name="$2"
    local max_wait=120
    local waited=0

    echo -n "Waiting for $name (port $port) "
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://localhost:$port/health" &>/dev/null; then
            echo " ✓ READY"
            return 0
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo " ✗ TIMEOUT (${max_wait}s)"
    return 1
}

cleanup() {
    echo ""
    echo "Stopping engine containers..."
    docker stop "$NNUNET_CONTAINER" "$NNUNET_OLD_CONTAINER" "$AUTOPET_CONTAINER" "$TOTALSEG_CONTAINER" 2>/dev/null || true
    docker rm "$NNUNET_CONTAINER" "$NNUNET_OLD_CONTAINER" "$AUTOPET_CONTAINER" "$TOTALSEG_CONTAINER" 2>/dev/null || true
    echo "Done."
}

# --- Main Execution ---
echo "============================================"
echo "  PET/CT App — Pull & Start"
echo "============================================"

trap cleanup EXIT

# Step 1: Pull images from DockerHub
echo -e "\n-- Step 1: Pulling images from DockerHub --"
pull_image "$NNUNET_IMAGE"
pull_image "$NNUNET_OLD_IMAGE"
pull_image "$AUTOPET_IMAGE"
pull_image "$TOTALSEG_IMAGE"

# Step 2: Start containers
echo -e "\n-- Step 2: Starting containers --"
start_container "$NNUNET_CONTAINER"     "$NNUNET_IMAGE"     "$NNUNET_PORT"
start_container "$NNUNET_OLD_CONTAINER" "$NNUNET_OLD_IMAGE" "$NNUNET_OLD_PORT"
start_container "$AUTOPET_CONTAINER"    "$AUTOPET_IMAGE"    "$AUTOPET_PORT"
start_container "$TOTALSEG_CONTAINER"   "$TOTALSEG_IMAGE"   "$TOTALSEG_PORT"

# Step 3: Health checks
echo -e "\n-- Step 3: Health checks --"
wait_for_health "$NNUNET_PORT"     "nnUNet Engine"
wait_for_health "$NNUNET_OLD_PORT" "nnUNet Old Engine"
wait_for_health "$AUTOPET_PORT"    "AutoPET Engine"
wait_for_health "$TOTALSEG_PORT"   "TotalSeg Engine"

# Step 4: Launch GUI
echo -e "\n-- Step 4: Launch PyQt GUI --"
cd "$SCRIPT_DIR"
uv run python -m src.main
