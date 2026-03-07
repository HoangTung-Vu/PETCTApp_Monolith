# PET/CT Segmentation Application

<div align="center">
  <h3>A desktop application for PET/CT image segmentation, refinement, and quantification.</h3>
</div>

## 📖 Project Overview

This project is a sophisticated desktop application built with Python, PyQt6, and Napari, designed for medical professionals and researchers to process PET/CT imaging data. It provides an end-to-end workflow from loading NIfTI files to running AI-driven segmentation, manual refinement, and generating clinical quantifications (like SUVmax, SUVmean, MTV, and gTLG).

Key features:
*   **Multi-Modal Visualization:** Interactive orthogonal (grid) views, fusion (overlay) layouts, and 3D volume rendering powered by Napari.
*   **Automated AI Segmentation:** Integrated deeply with Dockerized backends (nnUNet, TotalSegmentator) for rapid tumor and organ segmentation.
*   **Precision Refinement Tools:** Paint/Eraser 3D brushes, SUV-based thresholding, Iterative Thresholding, and AutoPET Interactive click-based probability refinement.
*   **Clinical Quantification:** Automatic calculation of lesion metrics and global Total Lesion Glycolysis (gTLG) with precise voxel volume formulation.
*   **Persistent Storage:** Local SQLite database integrated with SQLAlchemy for robust session management.

---

## 🏗️ Architecture Summary

The application is structured as a monolith encompassing independent components:

1.  **GUI Application (Host):** The frontend desktop app handles user interaction, rendering, and logic synchronization.
2.  **AI Engines (Docker):** To prevent dependency conflicts (e.g., PyTorch, CUDA), all AI engines run in isolated Docker containers:
    *   `engine-nnunet`: Baseline tumor segmentation.
    *   `engine-autopet`: Interactive tumor segmentation with user click integration.
    *   `engine-totalseg`: Organ segmentation.
3.  **Communication:** The GUI uses HTTP to send data and receive NIfTI masks or NPZ probability maps from the Docker engines.


---

## 💻 Installation & Setup

To run the AI models efficiently, an **NVIDIA GPU is highly recommended**.

### Option 1: Windows (Recommended for end-users)
Windows 10/11 makes handling Docker and GPUs incredibly easy thanks to WSL 2.

1.  **NVIDIA GPU Drivers:** Install standard NVIDIA Game Ready or Studio drivers.
2.  **Docker Desktop:** Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).
3.  **Configuration:** In Docker Desktop Settings > General, ensure **"Use the WSL 2 based engine"** is checked. The NVIDIA Container Toolkit is implicitly included.

### Option 2: Linux (Ubuntu/Debian)
On native Linux, you must manually install drivers and the toolkit.

1.  **NVIDIA GPU Drivers:**
    ```bash
    sudo ubuntu-drivers autoinstall
    sudo reboot
    ```
2.  **Docker Engine:** Follow [official Docker instructions](https://docs.docker.com/engine/install/ubuntu/).
3.  **NVIDIA Container Toolkit:** Install to allow Docker to use `--gpus all` (see [official docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
    ```bash
    # Example commands to configure repository and install
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

---

## 🚀 Running the Application

Ensure you have Python and `uv` installed to manage the virtual environment.

Run the launcher script from the root of the project:

```bash
# On Linux / WSL
./start.sh

# On Windows (Native)
.\start.bat
```

The script will automatically:
1. Load environment variables from `.env`.
2. Build the Docker images for the AI engines if they don't exist.
3. Start the containers (with GPU passthrough enabled).
4. Launch the PyQt6 GUI application.

---

## 📂 Project Structure

```text
PETCTApp_Monolith/
├── src/                     # Main PyQt app (GUI components, core logic, database models)
├── AI_engines/              # Dockerized AI engine backends (nnUNet, AutoPET, TotalSeg)
├── storage/                 # Data directory containing the SQLite database and NIfTI sessions
├── tests/                   # Test suite
├── start.sh / start.bat     # Launchers for the app and Docker backends
└── pyproject.toml           # Python dependencies (managed by uv)
```
