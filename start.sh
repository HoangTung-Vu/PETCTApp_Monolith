#!/bin/bash
# start.sh — Build nnUNet engine Docker image, run container, launch GUI (Linux dev).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
else
    echo "Warning: .env file not found, using defaults."
fi

# [1] Configuration
NNUNET_PORT="${ENGINE_NNUNET_PORT:-8104}"
NNUNET_IMAGE="${ENGINE_NNUNET_IMAGE:-nnunet-engine}"
NNUNET_CONTAINER="${ENGINE_NNUNET_CONTAINER:-nnunet-engine-container}"

# Detect logical CPU count for numpy/torch thread tuning
CPU_CORES="$(nproc 2>/dev/null || echo 4)"

GPU_FLAG=""
if /usr/bin/docker info 2>/dev/null | grep -i "Runtimes.*nvidia" >/dev/null 2>&1 || nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Enabling GPU passthrough."
    GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU detected. Running CPU mode (see README_SETUP.md for GPU)."
fi

# [2] Helper Functions

def_build_image() {
    local image_name="$1"
    local context_dir="$2"
    if /usr/bin/docker image inspect "$image_name" >/dev/null 2>&1; then
        read -r -p "Image '$image_name' already exists. Rebuild? [y/N] " answer
        case "$answer" in
            [yY][eE][sS]|[yY])
                echo "Pruning builder cache..."
                /usr/bin/docker builder prune -f
                echo "Rebuilding Docker image: $image_name ..."
                /usr/bin/docker build --no-cache -t "$image_name" "$context_dir"
                ;;
            *)
                echo "Skipping build for '$image_name', using existing image."
                ;;
        esac
    else
        echo "Pruning builder cache..."
        /usr/bin/docker builder prune -f
        echo "Building Docker image: $image_name ..."
        /usr/bin/docker build --no-cache -t "$image_name" "$context_dir"
    fi
}

start_container() {
    local container_name="$1"
    local image_name="$2"

    if /usr/bin/docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        /usr/bin/docker stop "$container_name" 2>/dev/null || true
        /usr/bin/docker rm "$container_name" 2>/dev/null || true
    fi

    echo "Starting container: $container_name (max resources, $CPU_CORES cores)"
    # --network host        : skip userland proxy, native loopback latency.
    # --ipc=host            : share host /dev/shm (supersedes --shm-size, no cap on PyTorch shm).
    # --ulimit memlock=-1   : unlimited pinned memory for CUDA DMA / fast H2D transfers.
    # --ulimit stack=67108864: 64MB stack (some torch ops need > default 8MB).
    # -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : reduces VRAM fragmentation.
    # -e PYTHONUNBUFFERED=1 : flush prints immediately (useful when tailing logs).
    # -e OMP/MKL/OPENBLAS_NUM_THREADS=$CPU_CORES : let CPU ops use all cores (preprocessing, resampling).
    /usr/bin/docker run -d \
        --name "$container_name" \
        $GPU_FLAG \
        --network host \
        --ipc=host \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e PYTHONUNBUFFERED=1 \
        -e OMP_NUM_THREADS="$CPU_CORES" \
        -e MKL_NUM_THREADS="$CPU_CORES" \
        -e OPENBLAS_NUM_THREADS="$CPU_CORES" \
        -e NUMEXPR_NUM_THREADS="$CPU_CORES" \
        "$image_name"
}

wait_for_health() {
    local port="$1"
    local name="$2"
    # Startup preloads model weights into VRAM; bump timeout from 60s to 180s.
    local max_wait=180
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
    echo "Stopping engine container..."
    /usr/bin/docker stop "$NNUNET_CONTAINER" 2>/dev/null || true
    /usr/bin/docker rm "$NNUNET_CONTAINER" 2>/dev/null || true
    echo "Done."
}

# [3] Main Execution
echo "====================================="
echo "  PET/CT Segmentation App"
echo "====================================="

trap cleanup EXIT

echo -e "\n-- Step 1: Building Docker image --"
def_build_image "$NNUNET_IMAGE" "$SCRIPT_DIR/AI_engines/engine_nnunet_old_ver"

echo -e "\n-- Step 2: Starting container --"
start_container "$NNUNET_CONTAINER" "$NNUNET_IMAGE"

echo -e "\n-- Step 3: Health check (waits for model preload) --"
wait_for_health "$NNUNET_PORT" "nnUNet Engine"

echo -e "\n-- Step 4: Launch PyQt GUI --"
cd "$SCRIPT_DIR"
uv run python -m src.main
