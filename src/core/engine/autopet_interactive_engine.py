import os
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.autopet_predictor import autoPETPredictor

from ..config import settings
from .base import SegmentationEngine


class AutoPETInteractiveEngine(SegmentationEngine):
    """Engine for promptable (click-based) PET/CT tumor segmentation 
    using the autoPET-interactive model.
    
    This wraps the autoPETPredictor which accepts CT+PET images and
    interactive click annotations (tumor/background points).
    
    Weights directory structure (model_dir):
        fold_0/ ... fold_9/
        dataset.json
        dataset_fingerprint.json
        plans.json
    """
    
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
        point_width: float = 2.0,
        use_folds: tuple = (0,),
        use_mirroring: bool = False,
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """
        Args:
            model_dir: Path to the model weights directory (fold_0..9, dataset.json, plans.json).
                       Defaults to settings.WEIGHTS_DIR / "autopet-interactive".
            device: "auto", "cuda", or "cpu".
            point_width: Radius for click point blobs (default 2).
            use_folds: Tuple of fold indices to ensemble.
            use_mirroring: Whether to use test-time mirroring augmentation.
            checkpoint_name: Name of checkpoint file in each fold directory.
        """
        if model_dir is None:
            self.model_dir = settings.WEIGHTS_DIR / "autopet-interactive"
        else:
            self.model_dir = Path(model_dir)
        
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.point_width = point_width
        self.use_folds = use_folds
        self.use_mirroring = use_mirroring
        self.checkpoint_name = checkpoint_name
        self.predictor = None
    
    def _init_predictor(self):
        """Lazily initialize the autoPETPredictor."""
        if self.predictor is not None:
            return
        
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please download the autoPET-interactive weights and place them there.\n"
                f"Expected structure: _model/fold_0..9, dataset.json, plans.json"
            )
        
        device = torch.device(self.device)
        
        # Handle memory fragmentation on CUDA
        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()
        
        self.predictor = autoPETPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=(self.device == "cuda"),
            device=device,
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        
        print(f"[AutoPET] Loading model from {self.model_dir}, folds={self.use_folds}")
        self.predictor.initialize_from_trained_model_folder(
            str(self.model_dir),
            use_folds=self.use_folds,
            checkpoint_name=self.checkpoint_name,
        )
        print("[AutoPET] Model loaded successfully")
    
    @staticmethod
    def _nib_to_sitk_array(img: nib.Nifti1Image) -> np.ndarray:
        """Convert nibabel image to SimpleITK-style array (Z, Y, X)."""
        # nibabel stores data as X, Y, Z; SimpleITK/nnUNet expects Z, Y, X
        arr = np.asanyarray(img.dataobj)
        return arr.transpose(2, 1, 0)
    
    @staticmethod
    def _get_spacing_zyx(img: nib.Nifti1Image) -> list:
        """Get spacing in Z, Y, X order (SimpleITK convention)."""
        spacing_xyz = img.header.get_zooms()[:3]
        return list(spacing_xyz[::-1])  # X,Y,Z -> Z,Y,X
    
    @staticmethod
    def _format_clicks(clicks: Optional[List[Dict[str, Any]]]) -> Dict:
        """Format clicks into the autoPET-interactive expected format.
        
        Input format (from GUI):
            [
                {"point": [z, y, x], "name": "tumor"},
                {"point": [z, y, x], "name": "background"},
                ...
            ]
        
        Output format (for autoPETPredictor):
            {"points": [{"point": [z, y, x], "name": "tumor"}, ...]}
        """
        if clicks is None:
            return {"points": []}
        
        # If already in the correct format with "points" key
        if isinstance(clicks, dict) and "points" in clicks:
            return clicks
        
        # If it's a list of click dicts
        if isinstance(clicks, list):
            return {"points": clicks}
        
        return {"points": []}
    
    def run(
        self,
        input_paths: Union[str, Path, List[Union[str, Path]]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
    ) -> nib.Nifti1Image:
        """Run interactive segmentation on CT + PET file paths with click prompts.
        
        Args:
            input_paths: List of [ct_path, pet_path] or a single path (CT only).
            clicks: Click annotations. List of dicts with 'point' and 'name' keys,
                    or dict with 'points' key. Coordinates are in [z, y, x] order.
        
        Returns:
            Segmentation result as Nifti1Image (in nibabel X,Y,Z space).
        """
        self._init_predictor()
        
        if isinstance(input_paths, (str, Path)):
            input_paths = [input_paths]
        input_paths = [Path(p) for p in input_paths]
        
        print(f"[AutoPET] Running inference on {[str(p) for p in input_paths]}")
        
        # Load images via SimpleITK (native format for nnUNet)
        arrays = []
        ref_sitk = None
        for p in input_paths:
            sitk_img = sitk.ReadImage(str(p))
            if ref_sitk is None:
                ref_sitk = sitk_img
            arrays.append(sitk.GetArrayFromImage(sitk_img))
        
        # Stack as channels: (C, Z, Y, X)
        input_array = np.stack(arrays)
        
        # Get spacing in Z,Y,X order
        spacing_zyx = list(ref_sitk.GetSpacing()[::-1])
        props = {'spacing': spacing_zyx}
        
        # Format clicks
        formatted_clicks = self._format_clicks(clicks)
        
        print(f"[AutoPET] Input shape: {input_array.shape}, spacing: {spacing_zyx}")
        print(f"[AutoPET] Clicks: {len(formatted_clicks.get('points', []))} points")
        
        # Run prediction
        ret = self.predictor.predict_single_npy_array(
            input_array, props, formatted_clicks, self.point_width, None, None, False
        )
        
        # Convert result back to nibabel format
        # ret is in Z,Y,X -> transpose to X,Y,Z for nibabel
        pred_array = np.transpose(ret, (2, 1, 0))
        
        # Use reference nibabel image for affine/header
        ref_nib = nib.load(str(input_paths[0]))
        
        print(f"[AutoPET] Output shape: {pred_array.shape}")
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_nib.affine, ref_nib.header)
    
    def run_nib(
        self,
        images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
    ) -> nib.Nifti1Image:
        """Run interactive segmentation on nibabel images with click prompts.
        
        Args:
            images: Single or list of Nifti1Image objects [CT, PET].
            clicks: Click annotations in [z, y, x] coordinate order.
        
        Returns:
            Segmentation result as Nifti1Image.
        """
        self._init_predictor()
        
        if isinstance(images, nib.Nifti1Image):
            images = [images]
        
        ref_img = images[0]
        print(f"[AutoPET] Running inference on {len(images)} nibabel image(s)")
        
        # Convert each image: X,Y,Z -> Z,Y,X for nnUNet
        arrays = [self._nib_to_sitk_array(img) for img in images]
        
        # Stack as channels: (C, Z, Y, X)
        input_array = np.stack(arrays)
        
        # Get spacing in Z,Y,X order
        spacing_zyx = self._get_spacing_zyx(ref_img)
        props = {'spacing': spacing_zyx}
        
        # Format clicks
        formatted_clicks = self._format_clicks(clicks)
        
        print(f"[AutoPET] Input shape: {input_array.shape}, spacing: {spacing_zyx}")
        print(f"[AutoPET] Clicks: {len(formatted_clicks.get('points', []))} points")
        
        # Run prediction
        ret = self.predictor.predict_single_npy_array(
            input_array, props, formatted_clicks, self.point_width, None, None, False
        )
        
        # Convert result back: Z,Y,X -> X,Y,Z for nibabel
        pred_array = np.transpose(ret, (2, 1, 0))
        
        print(f"[AutoPET] Output shape: {pred_array.shape}")
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)
