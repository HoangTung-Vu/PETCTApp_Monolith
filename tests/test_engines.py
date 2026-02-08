"""Test segmentation engines with PET/CT files in temp/"""
from pathlib import Path
import nibabel as nib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"

CT_PATH = TEMP_DIR / "ct.nii.gz"
PET_PATH = TEMP_DIR / "pet.nii.gz"
BODY_OUTPUT = TEMP_DIR / "body.nii.gz"
TUMOR_OUTPUT = TEMP_DIR / "tumor.nii.gz"


def test_totalseg():
    """Test TotalSegmentator with file path."""
    from src.core.engine import TotalSegEngine
    print(f"\n=== TotalSegmentator (file path) ===")
    engine = TotalSegEngine(task="total", fast=True)
    result = engine.run(CT_PATH)
    engine.save(result, BODY_OUTPUT)
    print(f"Labels: {set(result.get_fdata().flatten().astype(int))}")


def test_totalseg_nib():
    """Test TotalSegmentator with nibabel image."""
    from src.core.engine import TotalSegEngine
    print(f"\n=== TotalSegmentator (nibabel) ===")
    ct_img = nib.load(CT_PATH)
    print(f"[Test] Loaded CT, shape={ct_img.shape}")
    
    engine = TotalSegEngine(task="total", fast=True)
    result = engine.run_nib(ct_img)
    engine.save(result, BODY_OUTPUT)
    print(f"Labels: {set(result.get_fdata().flatten().astype(int))}")


def test_nnunet():
    """Test nnUNet with file paths."""
    from src.core.engine import NNUNetEngine
    print(f"\n=== nnUNet (file path) ===")
    engine = NNUNetEngine()
    result = engine.run([CT_PATH, PET_PATH])
    engine.save(result, TUMOR_OUTPUT)
    print(f"Labels: {set(result.get_fdata().flatten().astype(int))}")


def test_nnunet_nib():
    """Test nnUNet with nibabel images."""
    from src.core.engine import NNUNetEngine
    print(f"\n=== nnUNet (nibabel) ===")
    ct_img = nib.load(CT_PATH)
    pet_img = nib.load(PET_PATH)
    print(f"[Test] Loaded CT={ct_img.shape}, PET={pet_img.shape}")
    
    engine = NNUNetEngine()
    result = engine.run_nib([ct_img, pet_img])
    engine.save(result, TUMOR_OUTPUT)
    print(f"Labels: {set(result.get_fdata().flatten().astype(int))}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["totalseg", "nnunet", "both"], default="both")
    parser.add_argument("--mode", choices=["path", "nib", "both"], default="both", 
                        help="path=file path, nib=nibabel image")
    args = parser.parse_args()
    
    if args.engine in ("totalseg", "both"):
        if args.mode in ("path", "both"):
            test_totalseg()
        if args.mode in ("nib", "both"):
            test_totalseg_nib()
    
    if args.engine in ("nnunet", "both"):
        if args.mode in ("path", "both"):
            test_nnunet()
        if args.mode in ("nib", "both"):
            test_nnunet_nib()
