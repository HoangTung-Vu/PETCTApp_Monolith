from pathlib import Path
import os


# Base directory of this engine
BASE_DIR = Path(__file__).resolve().parent.parent

# Weights directory
WEIGHTS_DIR = BASE_DIR / "weights"

# Port
PORT = int(os.getenv("ENGINE_NNUNET_PORT", "8101"))


def setup_nnunet_env():
    """Set nnUNet environment variables pointing to this engine's weights."""
    nnunet_base = WEIGHTS_DIR
    
    raw_dir = nnunet_base / "nnUNet_raw"
    preprocessed_dir = nnunet_base / "nnUNet_preprocessed"
    results_dir = nnunet_base / "nnUNet_results"

    # Auto-create directories if not exist so iterdir() won't crash
    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["nnUNet_results"] = str(results_dir)
