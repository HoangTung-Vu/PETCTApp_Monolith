"""AutoPET Interactive Segmentation Engine — standalone version for Docker backend."""

import os
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import nibabel as nib
import numpy as np
import torch

from nnunetv2.inference.autopet_predictor import autoPETPredictor

from . import WEIGHTS_DIR


class AutoPETInteractiveEngine:
    """Engine for promptable (click-based) PET/CT tumor segmentation
    using the autoPET-interactive model.

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
        if model_dir is None:
            self.model_dir = WEIGHTS_DIR
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
                f"Expected structure: fold_0..9, dataset.json, plans.json"
            )

        device = torch.device(self.device)

        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()

        self.predictor = autoPETPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=True if self.device == "cuda" else False,
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
        arr = np.asanyarray(img.dataobj)
        return arr.transpose(2, 1, 0)

    @staticmethod
    def _get_spacing_zyx(img: nib.Nifti1Image) -> list:
        """Get spacing in Z, Y, X order."""
        spacing_xyz = img.header.get_zooms()[:3]
        return list(spacing_xyz[::-1])

    @staticmethod
    def _format_clicks(clicks: Optional[Union[List[Dict[str, Any]], Dict]]) -> Dict:
        """Format clicks into the autoPET-interactive expected format."""
        if clicks is None:
            return {"points": []}
        if isinstance(clicks, dict) and "points" in clicks:
            return clicks
        if isinstance(clicks, list):
            return {"points": clicks}
        return {"points": []}

    # ── run: accepts nib images + clicks ──

    def run(
        self,
        images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
    ) -> nib.Nifti1Image:
        """Run interactive segmentation on nibabel images with click prompts."""
        self._init_predictor()

        if isinstance(images, nib.Nifti1Image):
            images = [images]

        ref_img = images[0]
        print(f"[AutoPET] Running inference on {len(images)} image(s)")

        arrays = [self._nib_to_sitk_array(img) for img in images]
        input_array = np.stack(arrays)

        spacing_zyx = self._get_spacing_zyx(ref_img)
        props = {'spacing': spacing_zyx}
        formatted_clicks = self._format_clicks(clicks)

        print(f"[AutoPET] Input shape: {input_array.shape}, spacing: {spacing_zyx}")
        print(f"[AutoPET] Clicks: {len(formatted_clicks.get('points', []))} points")

        ret = self.predictor.predict_single_npy_array(
            input_array, props, formatted_clicks, self.point_width, None, None, False
        )

        pred_array = np.transpose(ret, (2, 1, 0))

        print(f"[AutoPET] Output shape: {pred_array.shape}")
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)

    # ── run_nib: alias ──

    def run_nib(
        self,
        images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
    ) -> nib.Nifti1Image:
        """Alias for run()."""
        return self.run(images, clicks=clicks)

    # ── run_prob ──

    def run_prob(
        self,
        images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
        single_channel: bool = True,
    ) -> np.ndarray:
        """Run interactive segmentation and return probability maps."""
        self._init_predictor()

        if isinstance(images, nib.Nifti1Image):
            images = [images]

        ref_img = images[0]
        print(f"[AutoPET] Running prob inference on {len(images)} image(s)")

        arrays = [self._nib_to_sitk_array(img) for img in images]
        input_array = np.stack(arrays)

        spacing_zyx = self._get_spacing_zyx(ref_img)
        props = {'spacing': spacing_zyx}
        formatted_clicks = self._format_clicks(clicks)

        print(f"[AutoPET] Input shape: {input_array.shape}, spacing: {spacing_zyx}")
        print(f"[AutoPET] Clicks: {len(formatted_clicks.get('points', []))} points")

        seg, prob = self.predictor.predict_single_npy_array(
            input_array, props, formatted_clicks, self.point_width, None, None, True
        )

        print(f"[AutoPET] Prob output shape: {prob.shape}, dtype: {prob.dtype}")

        if single_channel:
            if prob.shape[0] > 1:
                prob = prob[1]
            else:
                prob = prob[0]
            prob = np.transpose(prob, (2, 1, 0))
            print(f"[AutoPET] Single channel prob shape: {prob.shape}")
        else:
            prob = np.transpose(prob, (0, 3, 2, 1))
            print(f"[AutoPET] Multi-channel prob shape: {prob.shape}")

        return prob

    # ── run_nib_prob: alias ──

    def run_nib_prob(
        self,
        images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
        clicks: Optional[Union[List[Dict], Dict]] = None,
        single_channel: bool = True,
    ) -> np.ndarray:
        """Alias for run_prob()."""
        return self.run_prob(images, clicks=clicks, single_channel=single_channel)
