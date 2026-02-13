"""Test autoPET-interactive engine with PET/CT files in temp/"""
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"

CT_PATH = TEMP_DIR / "ct.nii.gz"
PET_PATH = TEMP_DIR / "pet.nii.gz"
OUTPUT_PATH = TEMP_DIR / "autopet_result.nii.gz"


# Sample clicks for testing — adjust coordinates based on actual data
# Format: list of dicts with "point" (z, y, x) and "name" ("tumor" or "background")
SAMPLE_CLICKS = [
    {"point": [150, 200, 200], "name": "tumor"},
    {"point": [150, 100, 100], "name": "background"},
]


def test_autopet_path():
    """Test AutoPET Interactive with file paths."""
    from src.core.engine import AutoPETInteractiveEngine
    print(f"\n=== AutoPET Interactive (file paths) ===")
    
    engine = AutoPETInteractiveEngine()
    result = engine.run([CT_PATH, PET_PATH], clicks=SAMPLE_CLICKS)
    engine.save(result, OUTPUT_PATH)
    
    labels = set(result.get_fdata().flatten().astype(int))
    print(f"Output shape: {result.shape}")
    print(f"Labels: {labels}")


def test_autopet_nib():
    """Test AutoPET Interactive with nibabel images."""
    from src.core.engine import AutoPETInteractiveEngine
    print(f"\n=== AutoPET Interactive (nibabel) ===")
    
    ct_img = nib.load(CT_PATH)
    pet_img = nib.load(PET_PATH)
    print(f"[Test] Loaded CT={ct_img.shape}, PET={pet_img.shape}")
    
    engine = AutoPETInteractiveEngine()
    result = engine.run_nib([ct_img, pet_img], clicks=SAMPLE_CLICKS)
    engine.save(result, OUTPUT_PATH)
    
    labels = set(result.get_fdata().flatten().astype(int))
    print(f"Output shape: {result.shape}")
    print(f"Labels: {labels}")


def test_autopet_no_clicks():
    """Test AutoPET Interactive without any clicks (empty prompt)."""
    from src.core.engine import AutoPETInteractiveEngine
    print(f"\n=== AutoPET Interactive (no clicks) ===")
    
    ct_img = nib.load(CT_PATH)
    pet_img = nib.load(PET_PATH)
    
    engine = AutoPETInteractiveEngine()
    result = engine.run_nib([ct_img, pet_img], clicks=None)
    engine.save(result, OUTPUT_PATH)
    
    labels = set(result.get_fdata().flatten().astype(int))
    print(f"Output shape: {result.shape}")
    print(f"Labels: {labels}")


def test_import_only():
    """Verify import works without needing weights."""
    from src.core.engine import AutoPETInteractiveEngine
    print(f"\n=== Import Test ===")
    engine = AutoPETInteractiveEngine()
    print(f"Engine created, model_dir={engine.model_dir}")
    print(f"Device={engine.device}, point_width={engine.point_width}")
    print("Import OK!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["path", "nib", "no_clicks", "import", "all"], 
                        default="import",
                        help="path=file paths, nib=nibabel, no_clicks=no annotations, import=import only")
    args = parser.parse_args()
    
    if args.mode in ("import", "all"):
        test_import_only()
    if args.mode in ("path", "all"):
        test_autopet_path()
    if args.mode in ("nib", "all"):
        test_autopet_nib()
    if args.mode in ("no_clicks", "all"):
        test_autopet_no_clicks()
