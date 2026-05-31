# PET/CT Segmentation App — Setup Guide

## Architecture Overview

The application consists of two components that communicate over HTTP:

*   **GUI Application (PyQt6):** Runs directly on the host. Handles file loading, visualization, and segmentation triggering.
*   **AI Engine (Docker):** One Docker container hosting a FastAPI backend for tumor segmentation:
    *   `engine_nnunet_old_ver`: nnU-Net tumor segmentation with patch-level inference progress streaming (port 8104).

Isolating the engine in Docker prevents dependency conflicts (PyTorch, CUDA, nnU-Net) on the host system.

---

## Model Weights

**Weights are not distributed in this repository.** Contact the project author to obtain them.

Place the weights at:
```
AI_engines/engine_nnunet_old_ver/weights/
├── nnUNet_results/
├── nnUNet_preprocessed/
└── nnUNet_raw/
```

The `start.sh` / `start.bat` launchers bake the weights into the Docker image at build time (`COPY weights/` in the Dockerfile). The image is rebuilt automatically when you first run the launcher, or when you choose "Rebuild" when prompted.

---

## GPU Driver & Docker Setup

To run the AI engine efficiently, an **NVIDIA GPU is highly recommended**.

### Option 1: Windows (Recommended for end-users)

Windows 10/11 with WSL 2 has built-in GPU passthrough — no extra toolkit needed.

#### Prerequisites
1.  **NVIDIA GPU Drivers:** Install Game Ready or Studio drivers from NVIDIA's website.
2.  **Docker Desktop:** Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).

#### Configuration
1.  Open Docker Desktop → **Settings** → **General**.
2.  Ensure **"Use the WSL 2 based engine"** is checked.
3.  *(Optional)* Settings → Resources → WSL Integration: enable for your default WSL distro.

The `NVIDIA Container Toolkit` is included in Docker Desktop's WSL2 backend. The `--gpus all` flag in `start.bat` will work automatically.

---

### Option 2: Linux (Ubuntu/Debian)

#### Step 1 — NVIDIA GPU Drivers
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
# Verify:
nvidia-smi
```

#### Step 2 — Docker Engine
Follow the [official Docker documentation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

#### Step 3 — NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Reference: [Official NVIDIA Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Running the Application

Once the prerequisites are met and weights are in place:

```bash
# Linux
./start.sh

# Windows
.\start.bat
```

The launcher will:
1. Load environment variables from `.env` (creates defaults if missing).
2. Build the Docker image for the nnU-Net engine (weights are baked in at build time).
3. Start the container with GPU passthrough if detected.
4. Launch the PyQt6 GUI.

The engine preloads model weights into VRAM on startup — the health check waits up to 180 seconds for it to become ready.
