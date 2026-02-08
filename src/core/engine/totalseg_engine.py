import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union
import nibabel as nib
import torch

from ..config import settings
from .base import SegmentationEngine


class TotalSegEngine(SegmentationEngine):
    def __init__(self, task: str = "total", fast: bool = True, fastest: bool = False, device: str = "auto"):
        self.task = task
        self.fast = fast
        self.fastest = fastest
        self.device = device
        
        if self.device == "auto":
            self.device = "gpu" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu" and not self.fast and not self.fastest:
            self.fast = True
        
        self.weights_path = settings.WEIGHTS_DIR / "totalsegmentator"
    
    def _run_subprocess(self, input_path: Path, output_path: Path) -> None:
        """Run TotalSegmentator in subprocess to avoid env var conflicts."""
        script = f'''
import os
os.environ["TOTALSEG_WEIGHTS_PATH"] = "{self.weights_path}"
os.environ["nnUNet_raw"] = "{self.weights_path}"
os.environ["nnUNet_preprocessed"] = "{self.weights_path}"
os.environ["nnUNet_results"] = "{self.weights_path}"

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

input_img = nib.load("{input_path}")
result = totalsegmentator(
    input_img,
    output=None,
    ml=True,
    task="{self.task}",
    fast={self.fast},
    fastest={self.fastest},
    device="{self.device}",
    quiet=False,
    verbose=True,
    statistics=False
)
nib.save(result, "{output_path}")
print("SUCCESS")
'''
        
        print(f"[TotalSeg] Running (task={self.task}, fast={self.fast}, device={self.device})")
        print(f"[TotalSeg] Weights path: {self.weights_path}")
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed with code {result.returncode}")
    
    def run(self, input_path: Union[str, Path]) -> nib.Nifti1Image:
        input_path = Path(input_path)
        
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            output_path = Path(tmp.name)
        
        print(f"[TotalSeg] Input: {input_path}")
        self._run_subprocess(input_path, output_path)
        
        output_img = nib.load(output_path)
        output_path.unlink()
        
        print(f"[TotalSeg] Done, shape={output_img.shape}")
        return output_img
    
    def run_nib(self, image: nib.Nifti1Image) -> nib.Nifti1Image:
        """Run segmentation on nibabel image directly.
        
        Saves to temp file first, then runs subprocess (needed due to env var conflicts).
        """
        # Save input to temp file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            input_path = Path(tmp.name)
        nib.save(image, input_path)
        
        # Create output temp file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            output_path = Path(tmp.name)
        
        print(f"[TotalSeg] Input from nib.Nifti1Image, shape={image.shape}")
        self._run_subprocess(input_path, output_path)
        
        # Cleanup input temp, load output
        input_path.unlink()
        output_img = nib.load(output_path)
        output_path.unlink()
        
        print(f"[TotalSeg] Done, shape={output_img.shape}")
        return output_img
