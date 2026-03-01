"""Mock NIfTI tests for nnUNet engine compatibility."""

import io
import numpy as np
import nibabel as nib
import pytest
from httpx import AsyncClient, ASGITransport

from main import app


def make_mock_nifti(shape=(64, 64, 64), dtype=np.float32) -> nib.Nifti1Image:
    """Create a mock NIfTI volume with random data and identity affine."""
    data = np.random.rand(*shape).astype(dtype)
    affine = np.eye(4)
    # Set voxel spacing to 1mm
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 1.0
    return nib.Nifti1Image(data, affine)


def nifti_to_bytes(img: nib.Nifti1Image) -> bytes:
    """Serialize a nibabel Nifti1Image to .nii.gz bytes."""
    bio = io.BytesIO()
    file_map = img.make_file_map({"image": bio, "header": bio})
    img.to_file_map(file_map)
    return bio.getvalue()


class TestMockNifti:
    """Test that nibabel and numpy work correctly with mock volumes."""

    def test_create_mock_nifti(self):
        img = make_mock_nifti()
        assert img.shape == (64, 64, 64)
        assert img.get_fdata().dtype == np.float64  # get_fdata returns float64
        assert np.allclose(img.affine, np.eye(4))

    def test_nifti_serialization_roundtrip(self):
        """Test that NIfTI can be serialized and deserialized."""
        original = make_mock_nifti(shape=(32, 32, 32))
        data_bytes = nifti_to_bytes(original)
        assert len(data_bytes) > 0

        # Deserialize
        fh = nib.FileHolder(fileobj=io.BytesIO(data_bytes))
        loaded = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        assert loaded.shape == (32, 32, 32)
        np.testing.assert_allclose(loaded.get_fdata(), original.get_fdata(), atol=1e-5)

    def test_axis_transpose(self):
        """Test X,Y,Z <-> Z,Y,X transpose for nnUNet compatibility."""
        img = make_mock_nifti(shape=(10, 20, 30))  # X=10, Y=20, Z=30
        arr = np.asanyarray(img.dataobj)
        assert arr.shape == (10, 20, 30)

        # Transpose to Z,Y,X for nnUNet
        zyx = arr.transpose(2, 1, 0)
        assert zyx.shape == (30, 20, 10)

        # Transpose back
        xyz = zyx.transpose(2, 1, 0)
        assert xyz.shape == (10, 20, 30)
        np.testing.assert_array_equal(xyz, arr)


class TestNNUNetImport:
    """Test that nnunetv2 can be imported."""

    def test_import_nnunetv2(self):
        import nnunetv2
        assert nnunetv2 is not None

    def test_import_predictor(self):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        assert nnUNetPredictor is not None


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.anyio
    async def test_health(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
