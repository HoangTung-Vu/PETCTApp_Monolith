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
    os.environ["nnUNet_raw"] = str(nnunet_base / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_base / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_base / "nnUNet_results")
