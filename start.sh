#!/bin/bash
# start.sh — Build engine Docker images, run containers, launch GUI
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
else
    echo "Warning: .env file not found, using defaults."
fi

# [1] Configuration
NNUNET_PORT="${ENGINE_NNUNET_PORT:-8101}"
AUTOPET_PORT="${ENGINE_AUTOPET_PORT:-8102}"
TOTALSEG_PORT="${ENGINE_TOTALSEG_PORT:-8103}"

NNUNET_IMAGE="${ENGINE_NNUNET_IMAGE:-engine-nnunet}"
AUTOPET_IMAGE="${ENGINE_AUTOPET_IMAGE:-engine-autopet}"
TOTALSEG_IMAGE="${ENGINE_TOTALSEG_IMAGE:-engine-totalseg}"

NNUNET_CONTAINER="${ENGINE_NNUNET_CONTAINER:-engine-nnunet-container}"
AUTOPET_CONTAINER="${ENGINE_AUTOPET_CONTAINER:-engine-autopet-container}"
TOTALSEG_CONTAINER="${ENGINE_TOTALSEG_CONTAINER:-engine-totalseg-container}"

GPU_FLAG=""
if docker info | grep -i "Runtimes.*nvidia" >/dev/null 2>&1 || nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Enabling GPU passthrough."
    GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU detected. Running CPU mode (see README_SETUP.md for GPU)."
fi

# [2] Helper Functions

def_build_image() {
    local image_name="$1"
    local context_dir="$2"
    echo "Building Docker image: $image_name ..."
    docker build -t "$image_name" "$context_dir"
}

start_container() {
    local container_name="$1"
    local image_name="$2"
    local port="$3"

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
    local max_wait=60
    local waited=0

    echo -n "Waiting for $name (port $port) "
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://localhost:$port/health" &>/dev/null; then
            echo " OKAY"
            return 0
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo " TIMEOUT"
    return 1
}

cleanup() {
    echo "Stopping engine containers..."
    docker stop "$NNUNET_CONTAINER" "$AUTOPET_CONTAINER" "$TOTALSEG_CONTAINER" 2>/dev/null || true
    docker rm "$NNUNET_CONTAINER" "$AUTOPET_CONTAINER" "$TOTALSEG_CONTAINER" 2>/dev/null || true
    echo "Done."
}

# [3] Main Execution
echo "====================================="
echo "  PET/CT Segmentation App Launcher"
echo "====================================="

trap cleanup EXIT

echo -e "\n-- Step 1: Building Docker images --"
def_build_image "$NNUNET_IMAGE" "$SCRIPT_DIR/engine_nnunet"
def_build_image "$AUTOPET_IMAGE" "$SCRIPT_DIR/engine_autopet"
def_build_image "$TOTALSEG_IMAGE" "$SCRIPT_DIR/engine_totalseg"

echo -e "\n-- Step 2: Starting containers --"
start_container "$NNUNET_CONTAINER" "$NNUNET_IMAGE" "$NNUNET_PORT"
start_container "$AUTOPET_CONTAINER" "$AUTOPET_IMAGE" "$AUTOPET_PORT"
start_container "$TOTALSEG_CONTAINER" "$TOTALSEG_IMAGE" "$TOTALSEG_PORT"

echo -e "\n-- Step 3: Health checks --"
wait_for_health "$NNUNET_PORT" "nnUNet Engine"
wait_for_health "$AUTOPET_PORT" "AutoPET Engine"
wait_for_health "$TOTALSEG_PORT" "TotalSeg Engine"

echo -e "\n-- Step 4: Launch PyQt GUI --"
cd "$SCRIPT_DIR"
uv run python -m src.main
