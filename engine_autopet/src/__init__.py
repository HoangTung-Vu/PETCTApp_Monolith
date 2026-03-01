from pathlib import Path
import os


# Base directory of this engine
BASE_DIR = Path(__file__).resolve().parent.parent

# Weights directory (autopet-interactive model weights: fold_0..9, dataset.json, plans.json)
WEIGHTS_DIR = BASE_DIR / "weights"

# Port
PORT = int(os.getenv("ENGINE_AUTOPET_PORT", "8102"))
