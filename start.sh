#!/bin/bash
# ─────────────────────────────────────────────────────────────
# start.sh — Build engine Docker images, run containers, launch GUI
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "Warning: .env file not found in $SCRIPT_DIR, using defaults."
fi

# ──── Configuration ────
NNUNET_PORT="${ENGINE_NNUNET_PORT:-8101}"
AUTOPET_PORT="${ENGINE_AUTOPET_PORT:-8102}"
TOTALSEG_PORT="${ENGINE_TOTALSEG_PORT:-8103}"

NNUNET_IMAGE="${ENGINE_NNUNET_IMAGE:-engine-nnunet}"
AUTOPET_IMAGE="${ENGINE_AUTOPET_IMAGE:-engine-autopet}"
TOTALSEG_IMAGE="${ENGINE_TOTALSEG_IMAGE:-engine-totalseg}"

NNUNET_CONTAINER="${ENGINE_NNUNET_CONTAINER:-engine-nnunet-container}"
AUTOPET_CONTAINER="${ENGINE_AUTOPET_CONTAINER:-engine-autopet-container}"
TOTALSEG_CONTAINER="${ENGINE_TOTALSEG_CONTAINER:-engine-totalseg-container}"

# GPU Configuration
GPU_FLAG=""
if docker info | grep -i "Runtimes.*nvidia" >/dev/null 2>&1 || nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Enabling GPU passthrough."
    GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU or Container Toolkit detected. Containers will run in CPU mode."
    echo "If you have an NVIDIA GPU, please refer to README_SETUP.md for installation instructions."
fi

# ──── Helper Functions ────

build_if_needed() {
    local image_name="$1"
    local context_dir="$2"

    if ! docker image inspect "$image_name" &>/dev/null; then
        echo "Building Docker image: $image_name ..."
        docker build -t "$image_name" "$context_dir"
    else
        echo "Image $image_name already exists."
    fi
}

start_container() {
    local container_name="$1"
    local image_name="$2"
    local port="$3"
    local weights_dir="$4"

    # Stop and remove if already running
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Stopping existing container: $container_name"
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
    fi

    echo "Starting container: $container_name (port $port)"
    docker run -d \
        --name "$container_name" \
        $GPU_FLAG \
        -p "$port:$port" \
        -v "$weights_dir:/app/weights:ro" \
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

# ──── Main ────

echo "═══════════════════════════════════════════"
echo "  PET/CT Segmentation App — Launcher"
echo "═══════════════════════════════════════════"

# Trap Ctrl+C to cleanup containers
trap cleanup EXIT

# 1. Build Docker images if they don't exist
echo ""
echo "── Step 1: Build Docker images ──"
build_if_needed "$NNUNET_IMAGE" "$SCRIPT_DIR/engine_nnunet"
build_if_needed "$AUTOPET_IMAGE" "$SCRIPT_DIR/engine_autopet"
build_if_needed "$TOTALSEG_IMAGE" "$SCRIPT_DIR/engine_totalseg"

# 2. Start containers with weights mounted
echo ""
echo "── Step 2: Start engine containers ──"

# Weights directories (from old storage/weights/ or engine-local weights/)
NNUNET_WEIGHTS="$SCRIPT_DIR/engine_nnunet/weights"
AUTOPET_WEIGHTS="$SCRIPT_DIR/engine_autopet/weights"
TOTALSEG_WEIGHTS="$SCRIPT_DIR/engine_totalseg/weights"

start_container "$NNUNET_CONTAINER" "$NNUNET_IMAGE" "$NNUNET_PORT" "$NNUNET_WEIGHTS"
start_container "$AUTOPET_CONTAINER" "$AUTOPET_IMAGE" "$AUTOPET_PORT" "$AUTOPET_WEIGHTS"
start_container "$TOTALSEG_CONTAINER" "$TOTALSEG_IMAGE" "$TOTALSEG_PORT" "$TOTALSEG_WEIGHTS"

# 3. Wait for health checks
echo ""
echo "── Step 3: Health checks ──"
wait_for_health "$NNUNET_PORT" "nnUNet Engine"
wait_for_health "$AUTOPET_PORT" "AutoPET Engine"
wait_for_health "$TOTALSEG_PORT" "TotalSeg Engine"

# 4. Launch PyQt GUI
echo ""
echo "── Step 4: Launch PyQt GUI ──"
echo "Starting GUI..."
cd "$SCRIPT_DIR"
uv run python src/main.py

echo "GUI closed. Cleaning up..."
