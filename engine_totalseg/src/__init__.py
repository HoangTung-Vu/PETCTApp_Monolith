from pathlib import Path
import os


# Base directory of this engine
BASE_DIR = Path(__file__).resolve().parent.parent

# Weights directory
WEIGHTS_DIR = BASE_DIR / "weights"

# Port
PORT = int(os.getenv("ENGINE_TOTALSEG_PORT", "8103"))


def setup_totalseg_env():
    """Set TotalSegmentator environment variables pointing to this engine's weights."""
    os.environ["TOTALSEG_WEIGHTS_PATH"] = str(WEIGHTS_DIR)
    # Also set nnUNet env vars to avoid conflicts
    os.environ["nnUNet_raw"] = str(WEIGHTS_DIR)
    os.environ["nnUNet_preprocessed"] = str(WEIGHTS_DIR)
    os.environ["nnUNet_results"] = str(WEIGHTS_DIR)
