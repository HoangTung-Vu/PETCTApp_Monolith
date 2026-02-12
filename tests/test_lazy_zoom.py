import sys
import unittest
from unittest.mock import MagicMock, patch
from PyQt6.QtWidgets import QApplication, QWidget

# Mock napari
sys.modules["napari"] = MagicMock()
sys.modules["napari.qt"] = MagicMock()

from src.gui.components.layout_manager import LayoutManager

class MockViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = MagicMock()
        self.viewer.layers = []
        self.viewer.dims = MagicMock() # For sync
        self.viewer.camera = MagicMock() # For sync
        
        # Attributes expected by LayoutManager
        self.LAYER_NAMES = {
            "ct": "CT Image",
            "pet": "PET Image",
            "tumor": "Tumor Mask",
            "organ": "Organ Mask"
        }
        
        # Methods
        self.load_image = MagicMock()
        self.load_mask = MagicMock()
        self.set_camera_view = MagicMock()
        self.set_3d_view = MagicMock()

class TestLazyZoom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    @patch('src.gui.components.layout_manager.ViewerWidget', new=MockViewerWidget)
    def setUp(self):
        self.layout_manager = LayoutManager()
        
    def test_lazy_loading(self):
        # 1. Load Data
        ct_data = "CT_DATA"
        pet_data = "PET_DATA"
        affine = "AFFINE"
        
        # Access 3D viewer (which is now a MockViewerWidget instance)
        viewer_3d = self.layout_manager.viewer_3d
        
        self.layout_manager.load_data(ct_data, pet_data, affine)
        
        # Verify 3D viewer NOT loaded
        viewer_3d.load_image.assert_not_called()
        self.assertFalse(self.layout_manager._is_3d_loaded)
        
        # 2. Switch to 3D
        self.layout_manager.set_view_mode("3d")
        
        # Verify 3D viewer LOADED
        viewer_3d.load_image.assert_any_call(ct_data, affine, "ct", "gray")
        viewer_3d.load_image.assert_any_call(pet_data, affine, "pet", "jet", opacity=0.7)
        self.assertTrue(self.layout_manager._is_3d_loaded)
        
        # 3. Switch away and back (Should NOT load again)
        viewer_3d.load_image.reset_mock()
        self.layout_manager.set_view_mode("grid")
        self.layout_manager.set_view_mode("3d")
        viewer_3d.load_image.assert_not_called()

    def test_zoom_reset(self):
        self.layout_manager.reset_zoom()
        
        # Verify reset_view called on grid viewer
        # Use implicit key access to get the widget instance
        grid_viewer = self.layout_manager.grid_viewers[(0,0)]
        grid_viewer.viewer.reset_view.assert_called()
        
        self.layout_manager.overlay_viewer.viewer.reset_view.assert_called()

if __name__ == "__main__":
    unittest.main()
