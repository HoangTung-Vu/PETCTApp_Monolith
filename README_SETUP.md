# PET/CT Segmentation App Setup Guide

This document outlines the architecture of the application and the necessary installation steps for both Windows and Linux, specifically focusing on enabling GPU acceleration for the AI inference engines.

## 🏗️ Architecture Overview

The application is built using a monolith-like repository containing independent components that communicate over HTTP:

*   **GUI Application (PyQt6):** The main user interface for loading NIfTI files, viewing scans, and triggering segmentation. Runs directly on the host machine.
*   **AI Engines (Docker Backends):** Three independent Docker containers hosting FastAPI backends for different segmentation tasks:
    *   `engine-nnunet`: Runs tumor segmentation.
    *   `engine-autopet`: Runs interactive tumor segmentation (with user clicks).
    *   `engine-totalseg`: Runs organ segmentation.
*   **Communication:** The GUI sends HTTP requests to the respective engine's REST API. The engines process the NIfTI volumes and return probability maps or binary masks as NIfTI or NPZ files.

This architecture ensures that complex Python dependencies (like PyTorch, CUDA, nnUNet) are isolated within Docker, preventing dependency conflicts on the host system.

---

## 💻 Installation & Setup

To run the AI models efficiently, an NVIDIA GPU is highly recommended. The setup process differs slightly depending on your operating system.

### Option 1: Windows (Recommended for end-users)

Windows 10/11 makes handling Docker and GPUs incredibly easy thanks to WSL 2 (Windows Subsystem for Linux), which has built-in GPU passthrough capabilities.

#### Prerequisites
1.  **NVIDIA GPU Drivers:** Install the standard NVIDIA Game Ready or Studio drivers on your Windows machine (download from the NVIDIA website or GeForce Experience).
2.  **Docker Desktop:** Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).

#### Configuration
1.  Open Docker Desktop.
2.  Go to **Settings** (gear icon) > **General**.
3.  Ensure that **"Use the WSL 2 based engine"** is checked.
4.  *(Optional but recommended)* Go to **Settings** > **Resources** > **WSL Integration** and ensure integration is enabled for your default WSL distro if you are using one.

**That's it!** The `NVIDIA Container Toolkit` is implicitly included in Docker Desktop's WSL2 backend. When you run `start.sh` (via Git Bash or WSL), the `--gpus all` flag will automatically work.

---

### Option 2: Linux (Ubuntu/Debian)

On native Linux, you must manually install the NVIDIA drivers and the NVIDIA Container Toolkit to allow Docker to access the GPU.

#### Prerequisites
1.  **NVIDIA GPU Drivers:**
    ```bash
    sudo ubuntu-drivers autoinstall
    sudo reboot
    ```
    *Verify with `nvidia-smi` after reboot.*
2.  **Docker Engine:** Install Docker following the [official Docker documentation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

#### Installing NVIDIA Container Toolkit
Run the following commands to install the toolkit so Docker can use `--gpus all`. (Reference: [Official NVIDIA Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

1.  **Configure the repository:**
    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

2.  **Update package lists and install:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```

3.  **Configure Docker to use the NVIDIA runtime:**
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    ```

4.  **Restart the Docker daemon:**
    ```bash
    sudo systemctl restart docker
    ```

**Validation:**
After installation, the `start.sh` script will automatically detect the toolkit and append the `--gpus all` flag when starting the engine containers.

---

## 🚀 Running the Application

Once the prerequisites are met, simply run the launcher script from the root of the project:

```bash
./start.sh
```

The script will automatically:
1. Load environment variables from `.env` (creates defaults if missing).
2. Build the Docker images for the engines if they don't exist.
3. Start the containers (with GPU passthrough if detected).
4. Launch the PyQt6 GUI.
