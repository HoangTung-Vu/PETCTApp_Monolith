"""Test NNUNetEngine run_prob and run_nib_prob with real PET/CT data in temp/"""
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"

CT_PATH = TEMP_DIR / "ct.nii.gz"
PET_PATH = TEMP_DIR / "pet.nii.gz"


def test_run_prob():
    """Test run_prob with file paths — returns np.ndarray probability maps."""
    from src.core.engine import NNUNetEngine

    print(f"\n{'='*60}")
    print(f"=== test_run_prob (file paths) ===")
    print(f"{'='*60}")

    # Print input info
    ct_img = nib.load(CT_PATH)
    pet_img = nib.load(PET_PATH)
    print(f"[Input] CT shape: {ct_img.shape}, dtype: {ct_img.get_data_dtype()}")
    print(f"[Input] PET shape: {pet_img.shape}, dtype: {pet_img.get_data_dtype()}")
    print(f"[Input] CT spacing: {ct_img.header.get_zooms()[:3]}")

    engine = NNUNetEngine()

    print("\n--- Testing single_channel=True (default) ---")
    start = time.time()
    prob = engine.run_prob([CT_PATH, PET_PATH])
    elapsed = time.time() - start
    print(f"[Result] shape: {prob.shape}, time: {elapsed:.2f}s")
    assert prob.ndim == 3, f"Expected 3D for single_channel=True, got {prob.ndim}D"

    print("\n--- Testing single_channel=False ---")
    start = time.time()
    prob_multi = engine.run_prob([CT_PATH, PET_PATH], single_channel=False)
    elapsed = time.time() - start
    print(f"[Result] shape: {prob_multi.shape}, time: {elapsed:.2f}s")
    assert prob_multi.ndim == 4, f"Expected 4D for single_channel=False, got {prob_multi.ndim}D"
    
    print(f"\n[PASS] test_run_prob")


def test_run_nib_prob():
    """Test run_nib_prob with nibabel images — returns np.ndarray probability maps."""
    from src.core.engine import NNUNetEngine

    print(f"\n{'='*60}")
    print(f"=== test_run_nib_prob (nibabel images) ===")
    print(f"{'='*60}")

    ct_img = nib.load(CT_PATH)
    pet_img = nib.load(PET_PATH)
    print(f"[Input] CT shape: {ct_img.shape}, dtype: {ct_img.get_data_dtype()}")
    print(f"[Input] PET shape: {pet_img.shape}, dtype: {pet_img.get_data_dtype()}")

    engine = NNUNetEngine()

    start = time.time()
    prob = engine.run_nib_prob([ct_img, pet_img])
    elapsed = time.time() - start

    print(f"\n[Result] type: {type(prob)}")
    print(f"[Result] shape: {prob.shape}")
    print(f"[Result] ndim: {prob.ndim}")
    print(f"[Result] dtype: {prob.dtype}")
    print(f"[Result] min: {prob.min():.6f}, max: {prob.max():.6f}")
    print(f"[Result] Time: {elapsed:.2f}s")

    # Per-class stats
    for c in range(prob.shape[0]):
        class_prob = prob[c]
        print(f"  Class {c}: shape={class_prob.shape}, "
              f"min={class_prob.min():.6f}, max={class_prob.max():.6f}, "
              f"mean={class_prob.mean():.6f}")

    # Verify probs sum to ~1 across classes at each voxel
    prob_sum = prob.sum(axis=0)
    print(f"\n[Check] prob.sum(axis=0): min={prob_sum.min():.6f}, max={prob_sum.max():.6f}")

    assert isinstance(prob, np.ndarray), "run_nib_prob should return np.ndarray"
    assert prob.ndim >= 3, f"Expected at least 3D, got {prob.ndim}D"
    print(f"\n[PASS] test_run_nib_prob")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test NNUNetEngine prob methods")
    parser.add_argument("--mode", choices=["path", "nib", "both"], default="both",
                        help="path=run_prob, nib=run_nib_prob, both=run both")
    args = parser.parse_args()

    if args.mode in ("path", "both"):
        test_run_prob()
    if args.mode in ("nib", "both"):
        test_run_nib_prob()

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")
