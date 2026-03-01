"""Mock NIfTI tests for AutoPET Interactive engine compatibility."""

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

    def test_nifti_serialization_roundtrip(self):
        original = make_mock_nifti(shape=(32, 32, 32))
        data_bytes = nifti_to_bytes(original)
        assert len(data_bytes) > 0

        fh = nib.FileHolder(fileobj=io.BytesIO(data_bytes))
        loaded = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        assert loaded.shape == (32, 32, 32)
        np.testing.assert_allclose(loaded.get_fdata(), original.get_fdata(), atol=1e-5)

    def test_axis_transpose_zyx(self):
        """Test X,Y,Z <-> Z,Y,X transpose for nnUNet compatibility."""
        img = make_mock_nifti(shape=(10, 20, 30))
        arr = np.asanyarray(img.dataobj)
        zyx = arr.transpose(2, 1, 0)
        assert zyx.shape == (30, 20, 10)
        xyz = zyx.transpose(2, 1, 0)
        np.testing.assert_array_equal(xyz, arr)


class TestAutoPETImport:
    """Test that autopet-interactive nnunetv2 fork can be imported."""

    def test_import_nnunetv2(self):
        import nnunetv2
        assert nnunetv2 is not None

    def test_import_autopet_predictor(self):
        from nnunetv2.inference.autopet_predictor import autoPETPredictor
        assert autoPETPredictor is not None


class TestClicksParsing:
    """Test click format handling."""

    def test_format_clicks_list(self):
        from src.engine import AutoPETInteractiveEngine
        clicks = [
            {"point": [10, 20, 30], "name": "tumor"},
            {"point": [40, 50, 60], "name": "background"},
        ]
        result = AutoPETInteractiveEngine._format_clicks(clicks)
        assert "points" in result
        assert len(result["points"]) == 2

    def test_format_clicks_none(self):
        from src.engine import AutoPETInteractiveEngine
        result = AutoPETInteractiveEngine._format_clicks(None)
        assert result == {"points": []}


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.anyio
    async def test_health(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
