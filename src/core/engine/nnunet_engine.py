import os
from pathlib import Path
from typing import List, Union
import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from ..config import settings
from .base import SegmentationEngine


class NNUNetEngine(SegmentationEngine):
    def __init__(self, dataset_id: int = 42, configuration: str = "3d_fullres", device: str = "auto"):
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.device = device
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._setup_env()
        self.predictor = None
    
    def _setup_env(self):
        base_path = settings.WEIGHTS_DIR / "nnunet"
        os.environ["nnUNet_raw"] = str(base_path / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(base_path / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(base_path / "nnUNet_results")
    
    def _find_model_folder(self) -> Path:
        results_dir = Path(os.environ["nnUNet_results"])
        for d in results_dir.iterdir():
            if d.name.startswith(f"Dataset{self.dataset_id:03d}"):
                model_folder = d / f"nnUNetTrainer__nnUNetPlans__{self.configuration}"
                if model_folder.exists():
                    return model_folder
        raise FileNotFoundError(f"Model folder for Dataset{self.dataset_id:03d} not found")
    
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
            perform_everything_on_device=self.device == "cuda",
            device=torch.device(self.device),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        model_folder = self._find_model_folder()
        folds = self._detect_folds(model_folder)
        print(f"[nnUNet] Loading model from {model_folder}, folds={folds}")
        self.predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=folds)
    
    def run(self, input_paths: Union[str, Path, List[Union[str, Path]]]) -> nib.Nifti1Image:
        self._init_predictor()
        
        if isinstance(input_paths, (str, Path)):
            input_paths = [input_paths]
        input_paths = [str(p) for p in input_paths]
        
        print(f"[nnUNet] Running inference on {input_paths}")
        ref_img = nib.load(input_paths[0])
        
        results = self.predictor.predict_from_files(
            list_of_lists_or_source_folder=[input_paths],
            output_folder_or_list_of_truncated_output_files=None,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2
        )
        
        pred_array = results[0]
        if pred_array.ndim == 4 and pred_array.shape[0] == 1:
            pred_array = pred_array[0]
        
        # nnUNet output is Z,Y,X -> transpose to X,Y,Z for nibabel
        pred_array = np.transpose(pred_array, (2, 1, 0))
        
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)
    
    def run_nib(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]]) -> nib.Nifti1Image:
        """Run segmentation on nibabel images directly.
        
        Uses predict_single_npy_array with proper axis transposition.
        According to nnUNet docs, nibabel uses X,Y,Z but nnUNet expects Z,Y,X.
        """
        self._init_predictor()
        
        if isinstance(images, nib.Nifti1Image):
            images = [images]
        
        ref_img = images[0]
        print(f"[nnUNet] Running inference on {len(images)} nibabel image(s)")
        
        # Stack all channels: transpose each from X,Y,Z to Z,Y,X for nnUNet
        img_arrays = []
        for img in images:
            arr = np.asanyarray(img.dataobj)
            arr = arr.transpose([2, 1, 0])  # X,Y,Z -> Z,Y,X
            img_arrays.append(arr)
        
        # Stack as channels: (C, Z, Y, X)
        stacked = np.stack(img_arrays, axis=0)
        
        # Create properties with reversed spacing (X,Y,Z -> Z,Y,X)
        spacing = ref_img.header.get_zooms()[:3]
        props = {'spacing': spacing[::-1]}  # reverse for nnUNet
        
        print(f"[nnUNet] Input shape: {stacked.shape}, spacing: {props['spacing']}")
        
        # Use predict_single_npy_array
        pred_array = self.predictor.predict_single_npy_array(
            stacked, props, None, None, False
        )
        
        if pred_array.ndim == 4 and pred_array.shape[0] == 1:
            pred_array = pred_array[0]
        
        # nnUNet output is Z,Y,X -> transpose back to X,Y,Z for nibabel
        pred_array = np.transpose(pred_array, (2, 1, 0))
        
        print(f"[nnUNet] Output shape: {pred_array.shape}")
        return nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)
