"""NNUNet Segmentation Engine — standalone version for Docker backend."""

import os
from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from . import WEIGHTS_DIR, setup_nnunet_env


class NNUNetEngine:
    def __init__(self, dataset_id: int = 42, configuration: str = "3d_fullres", device: str = "auto"):
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.device = device

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        setup_nnunet_env()
        self.predictor = None

    def _find_model_folder(self) -> Path:
        results_dir = Path(os.environ["nnUNet_results"])
        for d in results_dir.iterdir():
            if d.name.startswith(f"Dataset{self.dataset_id:03d}"):
                model_folder = d / f"nnUNetTrainer__nnUNetPlans__{self.configuration}"
                if model_folder.exists():
                    return model_folder
        raise FileNotFoundError(f"Model folder for Dataset{self.dataset_id:03d} not found in {results_dir}")

    def _detect_folds(self, model_folder: Path) -> tuple:
        if (model_folder / "fold_all").exists():
            return ("all",)
        folds = [int(f.name.split("_")[1]) for f in model_folder.iterdir()
                 if f.is_dir() and f.name.startswith("fold_")]
        return tuple(sorted(folds)) if folds else None

    def _init_predictor(self):
        if self.predictor is not None:
            return

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True if self.device == "cuda" else False,
            device=torch.device(self.device),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )

        model_folder = self._find_model_folder()
        folds = self._detect_folds(model_folder)
        print(f"[nnUNet] Loading model from {model_folder}, folds={folds}")
        self.predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=folds)

    # ── run: accepts NIfTI bytes (deserialized as nib images) ──

    def run(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]]) -> nib.Nifti1Image:
        """Run segmentation on nibabel images. Returns binary mask as Nifti1Image."""
        self._init_predictor()

        if isinstance(images, nib.Nifti1Image):
            images = [images]

        ref_img = images[0]
        print(f"[nnUNet] Running inference on {len(images)} image(s)")

        # Stack all channels: transpose each from X,Y,Z -> Z,Y,X for nnUNet
        img_arrays = []
        for img in images:
            arr = np.asanyarray(img.dataobj)
            arr = arr.transpose([2, 1, 0])  # X,Y,Z -> Z,Y,X
            img_arrays.append(arr)

        stacked = np.stack(img_arrays, axis=0)  # (C, Z, Y, X)

        spacing = ref_img.header.get_zooms()[:3]
        props = {'spacing': spacing[::-1]}

        print(f"[nnUNet] Input shape: {stacked.shape}, spacing: {props['spacing']}")

        pred_array = self.predictor.predict_single_npy_array(
            stacked, props, None, None, False
        )

        if pred_array.ndim == 4 and pred_array.shape[0] == 1:
            pred_array = pred_array[0]

        # Z,Y,X -> X,Y,Z for nibabel
        pred_array = np.transpose(pred_array, (2, 1, 0))

        print(f"[nnUNet] Output shape: {pred_array.shape}")
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)

    # ── run_nib: alias for run (both accept nib images over HTTP) ──

    def run_nib(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]]) -> nib.Nifti1Image:
        """Alias for run() — both accept nibabel images."""
        return self.run(images)

    # ── run_prob: returns probability maps ──

    def run_prob(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]], single_channel: bool = True) -> np.ndarray:
        """Run segmentation and return probability maps as numpy array."""
        self._init_predictor()

        if isinstance(images, nib.Nifti1Image):
            images = [images]

        ref_img = images[0]
        print(f"[nnUNet] Running prob inference on {len(images)} image(s)")

        img_arrays = []
        for img in images:
            arr = np.asanyarray(img.dataobj)
            arr = arr.transpose([2, 1, 0])
            img_arrays.append(arr)

        stacked = np.stack(img_arrays, axis=0)

        spacing = ref_img.header.get_zooms()[:3]
        props = {'spacing': spacing[::-1]}

        print(f"[nnUNet] Input shape: {stacked.shape}, spacing: {props['spacing']}")

        seg, prob = self.predictor.predict_single_npy_array(
            stacked, props, None, None, True
        )

        print(f"[nnUNet] Prob output shape: {prob.shape}, dtype: {prob.dtype}")

        if single_channel:
            if prob.shape[0] > 1:
                prob = prob[1]
            else:
                prob = prob[0]
            prob = np.transpose(prob, (2, 1, 0))
            print(f"[nnUNet] Single channel prob shape: {prob.shape}")
        else:
            prob = np.transpose(prob, (0, 3, 2, 1))
            print(f"[nnUNet] Multi-channel prob shape: {prob.shape}")

        return prob

    # ── run_nib_prob: alias for run_prob ──

    def run_nib_prob(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]], single_channel: bool = True) -> np.ndarray:
        """Alias for run_prob() — both accept nibabel images."""
        return self.run_prob(images, single_channel=single_channel)
