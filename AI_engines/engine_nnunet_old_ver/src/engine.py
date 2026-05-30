"""NNUNet Segmentation Engine — standalone version for Docker backend."""

import gc
import logging
import os
from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import torch

logger = logging.getLogger("engine.nnunet")

# PyTorch 2.6 compatibility for nnUNetv2 2.2.1 model loading
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from . import WEIGHTS_DIR, setup_nnunet_env


class NNUNetEngine:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        setup_nnunet_env()
        self.predictor = None

    def _find_model_folder(self) -> Path:
        results_dir = Path(os.environ["nnUNet_results"])
        model_folder = results_dir / self.model_name
        if model_folder.exists():
            return model_folder
        raise FileNotFoundError(f"Model folder not found at {model_folder}")

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
            use_mirroring=True,
            perform_everything_on_gpu=True if self.device == "cuda" else False,
            device=torch.device(self.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False  # progress flows via _progress_callback, not the terminal tqdm bar
        )

        model_folder = self._find_model_folder()
        folds = self._detect_folds(model_folder)
        logger.info("Loading model '%s' from %s, folds=%s", self.model_name, model_folder, folds)
        self.predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=folds, checkpoint_name='checkpoint_best.pth')

    def unload(self):
        """Free this engine's predictor + GPU weights so the pool can be rebuilt."""
        if self.predictor is None:
            return
        # Drop references to the network + loaded fold parameters, then let CUDA
        # reclaim the VRAM. Best-effort attribute clearing before dropping the predictor.
        for attr in ("network", "list_of_parameters"):
            try:
                setattr(self.predictor, attr, None)
            except Exception:
                pass
        self.predictor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.debug("Engine for '%s' unloaded", self.model_name)

    # ── run: accepts NIfTI bytes (deserialized as nib images) ──

    def run(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]],
            progress_callback=None) -> nib.Nifti1Image:
        """Run segmentation on nibabel images. Returns binary mask as Nifti1Image.

        ``progress_callback(done, total)`` is invoked once per inference patch
        (and once more at completion) so callers can surface live progress.
        """
        self._init_predictor()

        if isinstance(images, nib.Nifti1Image):
            images = [images]

        ref_img = images[0]
        logger.info("Running inference on %d image(s) [model=%s]", len(images), self.model_name)

        # Stack all channels: transpose each from X,Y,Z -> Z,Y,X for nnUNet
        img_arrays = []
        for img in images:
            arr = np.asanyarray(img.dataobj)
            arr = arr.transpose([2, 1, 0])  # X,Y,Z -> Z,Y,X
            img_arrays.append(arr)

        # nibabel applies scl_slope/scl_inter in float64 — force float32 to match conv weights.
        stacked = np.stack(img_arrays, axis=0).astype(np.float32, copy=False)  # (C, Z, Y, X)

        spacing = ref_img.header.get_zooms()[:3]
        props = {'spacing': spacing[::-1]}

        logger.debug("Input shape: %s, dtype: %s, spacing: %s", stacked.shape, stacked.dtype, props['spacing'])

        self.predictor._progress_callback = progress_callback
        try:
            pred_array = self.predictor.predict_single_npy_array(
                stacked, props, None, None, False
            )
        finally:
            self.predictor._progress_callback = None

        if pred_array.ndim == 4 and pred_array.shape[0] == 1:
            pred_array = pred_array[0]

        # Z,Y,X -> X,Y,Z for nibabel
        pred_array = np.transpose(pred_array, (2, 1, 0))

        logger.debug("Output shape: %s", pred_array.shape)
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
        logger.info("Running prob inference on %d image(s) [model=%s]", len(images), self.model_name)

        img_arrays = []
        for img in images:
            arr = np.asanyarray(img.dataobj)
            arr = arr.transpose([2, 1, 0])
            img_arrays.append(arr)

        stacked = np.stack(img_arrays, axis=0).astype(np.float32, copy=False)

        spacing = ref_img.header.get_zooms()[:3]
        props = {'spacing': spacing[::-1]}

        logger.debug("Input shape: %s, dtype: %s, spacing: %s", stacked.shape, stacked.dtype, props['spacing'])

        seg, prob = self.predictor.predict_single_npy_array(
            stacked, props, None, None, True
        )

        logger.debug("Prob output shape: %s, dtype: %s", prob.shape, prob.dtype)

        if single_channel:
            if prob.shape[0] > 1:
                prob = prob[1]
            else:
                prob = prob[0]
            prob = np.transpose(prob, (2, 1, 0))
            logger.debug("Single channel prob shape: %s", prob.shape)
        else:
            prob = np.transpose(prob, (0, 3, 2, 1))
            logger.debug("Multi-channel prob shape: %s", prob.shape)

        return prob

    # ── run_nib_prob: alias for run_prob ──

    def run_nib_prob(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]], single_channel: bool = True) -> np.ndarray:
        """Alias for run_prob() — both accept nibabel images."""
        return self.run_prob(images, single_channel=single_channel)
