"""
Test suite for the probability-based refinement system.
Uses mock CT/PET data to test:
  1. FileManager numpy storage methods
  2. SessionManager tumor_prob persistence
  3. NNUNetEngine run_prob / run_nib_prob (mocked predictor)
  4. AutoPETInteractiveEngine run_prob / run_nib_prob (mocked predictor)
  5. Probability combining logic (averaging + thresholding)
"""
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import nibabel as nib
import numpy as np

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──── Helpers ────

def make_mock_nifti(shape=(64, 64, 32), dtype=np.float32, value_range=(0, 1000)):
    """Create a mock NIfTI image with random data."""
    rng = np.random.default_rng(42)
    data = rng.uniform(value_range[0], value_range[1], size=shape).astype(dtype)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])  # 2mm x 2mm x 3mm spacing
    return nib.Nifti1Image(data, affine)


def make_mock_prob(shape=(64, 64, 32)):
    """Create a mock probability volume (values 0-1)."""
    rng = np.random.default_rng(123)
    return rng.uniform(0, 1, size=shape).astype(np.float32)


# ════════════════════════════════════════════════════════════════
# 1. FileManager numpy storage
# ════════════════════════════════════════════════════════════════

class TestFileManagerNumpy:
    """Tests for FileManager.save_numpy / load_numpy / numpy_exists."""

    def setup_method(self):
        """Use a temporary directory for storage."""
        self.tmpdir = tempfile.mkdtemp()
        self._orig_data_dir = None

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_data_dir(self):
        from src.core.config import settings
        self._orig_data_dir = settings.DATA_DIR
        settings.DATA_DIR = Path(self.tmpdir)

    def _restore_data_dir(self):
        if self._orig_data_dir is not None:
            from src.core.config import settings
            settings.DATA_DIR = self._orig_data_dir

    def test_save_load_numpy(self):
        self._patch_data_dir()
        try:
            from src.core.file_manager import FileManager

            prob = make_mock_prob((64, 64, 32))
            session_id = 9999

            # Save
            path = FileManager.save_numpy(prob, session_id, "tumor_prob")
            assert path.exists(), f"File was not created at {path}"
            assert path.suffix == ".npy"

            # Load
            loaded = FileManager.load_numpy(session_id, "tumor_prob")
            assert loaded.shape == prob.shape, f"Shape mismatch: {loaded.shape} vs {prob.shape}"
            assert loaded.dtype == prob.dtype, f"Dtype mismatch: {loaded.dtype} vs {prob.dtype}"
            np.testing.assert_array_almost_equal(loaded, prob)
            print("[PASS] test_save_load_numpy")
        finally:
            self._restore_data_dir()

    def test_numpy_exists(self):
        self._patch_data_dir()
        try:
            from src.core.file_manager import FileManager

            session_id = 9998
            assert not FileManager.numpy_exists(session_id, "tumor_prob")

            prob = make_mock_prob((10, 10, 10))
            FileManager.save_numpy(prob, session_id, "tumor_prob")
            assert FileManager.numpy_exists(session_id, "tumor_prob")
            print("[PASS] test_numpy_exists")
        finally:
            self._restore_data_dir()

    def test_load_nonexistent_raises(self):
        self._patch_data_dir()
        try:
            from src.core.file_manager import FileManager

            try:
                FileManager.load_numpy(1, "tumor_prob")
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass
            print("[PASS] test_load_nonexistent_raises")
        finally:
            self._restore_data_dir()


# ════════════════════════════════════════════════════════════════
# 2. SessionManager tumor_prob
# ════════════════════════════════════════════════════════════════

class TestSessionManagerProb:
    """Tests for SessionManager.set_tumor_prob / get_tumor_prob / save / load."""

    def test_set_get_tumor_prob(self):
        from src.core.session_manager import SessionManager

        sm = SessionManager()
        assert sm.get_tumor_prob() is None

        prob = make_mock_prob((64, 64, 32))
        sm.set_tumor_prob(prob)

        result = sm.get_tumor_prob()
        assert result is not None
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, prob.astype(np.float32))
        print("[PASS] test_set_get_tumor_prob")

    def test_tumor_prob_save_load(self):
        """Test that tumor_prob persists through save/load cycle."""
        tmpdir = tempfile.mkdtemp()
        try:
            from src.core.config import settings
            from src.core.file_manager import FileManager
            from src.core.session_manager import SessionManager

            orig_data_dir = settings.DATA_DIR
            settings.DATA_DIR = Path(tmpdir)

            sm = SessionManager()
            sm.current_session_id = 7777

            # Set CT (needed for mask affine)
            ct_img = make_mock_nifti()
            sm.ct_image = ct_img

            # Set tumor mask + prob
            mask_data = np.zeros((64, 64, 32), dtype=np.uint8)
            mask_data[10:20, 10:20, 5:15] = 1
            sm.set_tumor_mask(mask_data)

            prob = make_mock_prob((64, 64, 32))
            sm.set_tumor_prob(prob)

            # Mock repository.update to avoid DB dependency
            sm.repository = MagicMock()

            # Save
            sm.save_session()

            # Verify files exist
            assert FileManager.numpy_exists(7777, "tumor_prob"), "tumor_prob.npy not saved"
            assert FileManager.file_exists(7777, "tumor_seg"), "tumor_seg.nii.gz not saved"

            # Create new SessionManager and load
            sm2 = SessionManager()
            sm2.current_session_id = 7777
            sm2.repository = MagicMock()
            sm2.repository.get_by_id.return_value = MagicMock()

            # Load session (only loads files that exist)
            if FileManager.file_exists(7777, "ct"):
                sm2.ct_image = FileManager.load_nifti(7777, "ct")
            if FileManager.file_exists(7777, "tumor_seg"):
                sm2.tumor_mask = FileManager.load_nifti(7777, "tumor_seg")
            if FileManager.numpy_exists(7777, "tumor_prob"):
                sm2.tumor_prob = FileManager.load_numpy(7777, "tumor_prob")

            # Verify
            assert sm2.tumor_prob is not None, "tumor_prob not loaded"
            assert sm2.tumor_prob.shape == prob.shape
            np.testing.assert_array_almost_equal(sm2.tumor_prob, prob.astype(np.float32))
            print("[PASS] test_tumor_prob_save_load")
        finally:
            settings.DATA_DIR = orig_data_dir
            shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
# 3. NNUNetEngine run_prob / run_nib_prob (mocked predictor)
# ════════════════════════════════════════════════════════════════

class TestNNUNetEngineProb:
    """Tests for NNUNetEngine.run_prob / run_nib_prob with mocked predictor.
    Verifies both shape AND value correctness after class selection + transposition.
    """

    def _create_engine_with_mock(self, seed=42):
        from src.core.engine.nnunet_engine import NNUNetEngine

        engine = NNUNetEngine(dataset_id=42, device="cpu")
        mock_predictor = MagicMock()

        # Deterministic mock: (num_classes=2, Z=32, Y=64, X=64)
        rng = np.random.default_rng(seed)
        num_classes, Z, Y, X = 2, 32, 64, 64
        seg = np.zeros((Z, Y, X), dtype=np.uint8)
        raw_prob = rng.uniform(0, 1, (num_classes, Z, Y, X)).astype(np.float32)
        # Normalize so probs sum to 1 across classes
        raw_prob = raw_prob / raw_prob.sum(axis=0, keepdims=True)

        mock_predictor.predict_single_npy_array.return_value = (seg, raw_prob)
        engine.predictor = mock_predictor
        return engine, raw_prob

    def test_run_nib_prob_single_channel_values(self):
        """Verify single_channel=True returns prob[1] transposed correctly."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        result = engine.run_nib_prob([ct_img, pet_img], single_channel=True)

        # raw_prob[1] has shape (Z=32, Y=64, X=64)
        # After class selection: (Z=32, Y=64, X=64)
        # After transpose (2,1,0): (X=64, Y=64, Z=32)
        expected = np.transpose(raw_prob[1], (2, 1, 0))

        assert result.shape == (64, 64, 32), f"Shape mismatch: {result.shape}"
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Spot-check some specific voxel values
        # result[x, y, z] should equal raw_prob[1, z, y, x]
        for x, y, z in [(0, 0, 0), (10, 20, 5), (63, 63, 31), (32, 32, 16)]:
            expected_val = raw_prob[1, z, y, x]
            actual_val = result[x, y, z]
            assert abs(actual_val - expected_val) < 1e-6, \
                f"Voxel ({x},{y},{z}): expected {expected_val}, got {actual_val}"

        # Value range check
        assert result.min() >= 0.0 and result.max() <= 1.0
        print(f"[PASS] test_run_nib_prob_single_channel_values")

    def test_run_nib_prob_multi_channel_values(self):
        """Verify single_channel=False returns all classes transposed correctly."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        result = engine.run_nib_prob([ct_img, pet_img], single_channel=False)

        # raw_prob shape: (2, Z=32, Y=64, X=64)
        # After transpose(0,3,2,1): (2, X=64, Y=64, Z=32)
        expected = np.transpose(raw_prob, (0, 3, 2, 1))

        assert result.shape == (2, 64, 64, 32), f"Shape mismatch: {result.shape}"
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Verify probs sum to ~1 across classes at each voxel
        prob_sum = result.sum(axis=0)
        np.testing.assert_array_almost_equal(prob_sum, np.ones((64, 64, 32)), decimal=5)

        # Spot-check: result[c, x, y, z] == raw_prob[c, z, y, x]
        for c, x, y, z in [(0, 5, 10, 15), (1, 5, 10, 15), (0, 63, 63, 31)]:
            expected_val = raw_prob[c, z, y, x]
            actual_val = result[c, x, y, z]
            assert abs(actual_val - expected_val) < 1e-6, \
                f"Voxel c={c},({x},{y},{z}): expected {expected_val}, got {actual_val}"
        print(f"[PASS] test_run_nib_prob_multi_channel_values")

    def test_run_nib_prob_class_selection_correct(self):
        """Verify single_channel selects class 1 (lesion), NOT class 0 (bg)."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        single = engine.run_nib_prob([ct_img, pet_img], single_channel=True)
        multi = engine.run_nib_prob([ct_img, pet_img], single_channel=False)

        # single should equal multi[1] (lesion class), NOT multi[0] (background)
        np.testing.assert_array_almost_equal(single, multi[1], decimal=6)

        # And should NOT equal background class
        assert not np.allclose(single, multi[0]), \
            "single_channel result should NOT equal background class"
        print(f"[PASS] test_run_nib_prob_class_selection_correct")

    def test_run_prob_file_path_values(self):
        """Test run_prob with file paths — verify values match run_nib_prob."""
        engine, raw_prob = self._create_engine_with_mock(seed=99)
        tmpdir = tempfile.mkdtemp()
        try:
            ct_img = make_mock_nifti((64, 64, 32))
            pet_img = make_mock_nifti((64, 64, 32))
            ct_path = Path(tmpdir) / "ct.nii.gz"
            pet_path = Path(tmpdir) / "pet.nii.gz"
            nib.save(ct_img, ct_path)
            nib.save(pet_img, pet_path)

            result = engine.run_prob([ct_path, pet_path], single_channel=True)

            # Expected: raw_prob[1] transposed (Z,Y,X) -> (X,Y,Z)
            expected = np.transpose(raw_prob[1], (2, 1, 0))
            assert result.shape == expected.shape
            np.testing.assert_array_almost_equal(result, expected, decimal=6)
            print(f"[PASS] test_run_prob_file_path_values")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
# 4. AutoPETInteractiveEngine run_prob / run_nib_prob (mocked)
# ════════════════════════════════════════════════════════════════

class TestAutoPETEngineProb:
    """Tests for AutoPETInteractiveEngine.run_prob / run_nib_prob with mocked predictor.
    Verifies both shape AND value correctness.
    """

    def _create_engine_with_mock(self, seed=77):
        from src.core.engine.autopet_interactive_engine import AutoPETInteractiveEngine

        engine = AutoPETInteractiveEngine(device="cpu")
        mock_predictor = MagicMock()

        rng = np.random.default_rng(seed)
        num_classes, Z, Y, X = 2, 32, 64, 64
        seg = np.zeros((Z, Y, X), dtype=np.uint8)
        raw_prob = rng.uniform(0, 1, (num_classes, Z, Y, X)).astype(np.float32)
        raw_prob = raw_prob / raw_prob.sum(axis=0, keepdims=True)

        mock_predictor.predict_single_npy_array.return_value = (seg, raw_prob)
        engine.predictor = mock_predictor
        return engine, raw_prob

    def test_run_nib_prob_single_channel_values(self):
        """Verify single_channel=True returns prob[1] transposed correctly."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))
        clicks = [{"point": [16, 32, 32], "name": "tumor"}]

        result = engine.run_nib_prob([ct_img, pet_img], clicks=clicks, single_channel=True)

        expected = np.transpose(raw_prob[1], (2, 1, 0))
        assert result.shape == (64, 64, 32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Spot-check: result[x, y, z] == raw_prob[1, z, y, x]
        for x, y, z in [(0, 0, 0), (10, 20, 5), (63, 63, 31)]:
            expected_val = raw_prob[1, z, y, x]
            actual_val = result[x, y, z]
            assert abs(actual_val - expected_val) < 1e-6, \
                f"Voxel ({x},{y},{z}): expected {expected_val}, got {actual_val}"

        assert result.min() >= 0.0 and result.max() <= 1.0
        print(f"[PASS] test_run_nib_prob_single_channel_values (AutoPET)")

    def test_run_nib_prob_multi_channel_values(self):
        """Verify multi-channel returns all classes with correct values."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        result = engine.run_nib_prob([ct_img, pet_img], clicks=None, single_channel=False)

        expected = np.transpose(raw_prob, (0, 3, 2, 1))
        assert result.shape == (2, 64, 64, 32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Probs sum to 1
        prob_sum = result.sum(axis=0)
        np.testing.assert_array_almost_equal(prob_sum, np.ones((64, 64, 32)), decimal=5)

        # Spot-check
        for c, x, y, z in [(0, 5, 10, 15), (1, 5, 10, 15)]:
            expected_val = raw_prob[c, z, y, x]
            actual_val = result[c, x, y, z]
            assert abs(actual_val - expected_val) < 1e-6
        print(f"[PASS] test_run_nib_prob_multi_channel_values (AutoPET)")

    def test_run_nib_prob_class_selection_correct(self):
        """Verify single_channel selects class 1 (lesion), NOT class 0 (bg)."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        single = engine.run_nib_prob([ct_img, pet_img], clicks=None, single_channel=True)
        multi = engine.run_nib_prob([ct_img, pet_img], clicks=None, single_channel=False)

        np.testing.assert_array_almost_equal(single, multi[1], decimal=6)
        assert not np.allclose(single, multi[0])
        print(f"[PASS] test_run_nib_prob_class_selection_correct (AutoPET)")

    def test_run_nib_prob_no_clicks_values(self):
        """Test with no clicks — values should still be correct."""
        engine, raw_prob = self._create_engine_with_mock()
        ct_img = make_mock_nifti((64, 64, 32))
        pet_img = make_mock_nifti((64, 64, 32))

        result = engine.run_nib_prob([ct_img, pet_img], clicks=None, single_channel=True)

        expected = np.transpose(raw_prob[1], (2, 1, 0))
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        print(f"[PASS] test_run_nib_prob_no_clicks_values (AutoPET)")


# ════════════════════════════════════════════════════════════════
# 5. Probability combining logic
# ════════════════════════════════════════════════════════════════

class TestProbCombining:
    """Tests for the probability averaging and thresholding logic."""

    def test_combine_with_existing_prob(self):
        """Average old + new prob and threshold."""
        shape = (64, 64, 32)
        old_prob = np.full(shape, 0.8, dtype=np.float32)   # high confidence
        new_prob = np.full(shape, 0.3, dtype=np.float32)   # low confidence

        combined = (old_prob + new_prob) / 2.0  # = 0.55
        mask = (combined >= 0.5).astype(np.uint8)

        assert np.all(mask == 1), "0.55 >= 0.5 should produce all-1 mask"
        np.testing.assert_almost_equal(combined, 0.55)
        print("[PASS] test_combine_with_existing_prob")

    def test_combine_no_existing_prob(self):
        """When no old prob, combined = new prob."""
        shape = (64, 64, 32)
        new_prob = make_mock_prob(shape)

        old_prob = None
        if old_prob is not None:
            combined = (old_prob + new_prob) / 2.0
        else:
            combined = new_prob

        mask = (combined >= 0.5).astype(np.uint8)
        expected_mask = (new_prob >= 0.5).astype(np.uint8)
        np.testing.assert_array_equal(mask, expected_mask)
        print("[PASS] test_combine_no_existing_prob")

    def test_threshold_boundary(self):
        """At exactly 0.5, voxel should be included."""
        shape = (10, 10, 10)
        prob = np.full(shape, 0.5, dtype=np.float32)
        mask = (prob >= 0.5).astype(np.uint8)
        assert np.all(mask == 1), "Exactly 0.5 should be included"

        prob_below = np.full(shape, 0.499, dtype=np.float32)
        mask_below = (prob_below >= 0.5).astype(np.uint8)
        assert np.all(mask_below == 0), "Below 0.5 should be excluded"
        print("[PASS] test_threshold_boundary")

    def test_combine_spatial_variation(self):
        """Verify combining works correctly with spatially varying probs."""
        shape = (10, 10, 10)
        # Old: high prob in first half, low in second half
        old_prob = np.zeros(shape, dtype=np.float32)
        old_prob[:5, :, :] = 0.9
        old_prob[5:, :, :] = 0.1

        # New: uniform 0.6
        new_prob = np.full(shape, 0.6, dtype=np.float32)

        combined = (old_prob + new_prob) / 2.0
        # First half: (0.9 + 0.6) / 2 = 0.75 -> 1
        # Second half: (0.1 + 0.6) / 2 = 0.35 -> 0
        mask = (combined >= 0.5).astype(np.uint8)

        assert np.all(mask[:5, :, :] == 1), "First half should be 1"
        assert np.all(mask[5:, :, :] == 0), "Second half should be 0"
        print("[PASS] test_combine_spatial_variation")


# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

def run_all():
    print(f"\n{'='*60}")
    print("=== Probability System Test Suite ===")
    print(f"{'='*60}\n")

    test_classes = [
        TestFileManagerNumpy,
        TestSessionManagerProb,
        TestNNUNetEngineProb,
        TestAutoPETEngineProb,
        TestProbCombining,
    ]

    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n── {cls.__name__} ──")
        obj = cls()
        for method_name in sorted(dir(obj)):
            if method_name.startswith("test_"):
                total += 1
                if hasattr(obj, "setup_method"):
                    obj.setup_method()
                try:
                    getattr(obj, method_name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    print(f"[FAIL] {method_name}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if hasattr(obj, "teardown_method"):
                        obj.teardown_method()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
