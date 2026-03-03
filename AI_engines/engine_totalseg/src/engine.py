"""TotalSegmentator Segmentation Engine — standalone version for Docker backend."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

import nibabel as nib
import torch

from . import WEIGHTS_DIR, setup_totalseg_env


class TotalSegEngine:
    def __init__(self, task: str = "total", fast: bool = True, fastest: bool = False, device: str = "auto"):
        self.task = task
        self.fast = fast
        self.fastest = fastest
        self.device = device

        if self.device == "auto":
            self.device = "gpu" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu" and not self.fast and not self.fastest:
            self.fast = True

        setup_totalseg_env()

    def _run_subprocess(self, input_path: Path, output_path: Path) -> None:
        """Run TotalSegmentator in subprocess to avoid env var conflicts."""
        script = f'''
import os
os.environ["TOTALSEG_WEIGHTS_PATH"] = "{WEIGHTS_DIR}"
os.environ["nnUNet_raw"] = "{WEIGHTS_DIR}"
os.environ["nnUNet_preprocessed"] = "{WEIGHTS_DIR}"
os.environ["nnUNet_results"] = "{WEIGHTS_DIR}"

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
        print(f"[TotalSeg] Weights path: {WEIGHTS_DIR}")

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed with code {result.returncode}")

    # ── run: accepts nib image ──

    def run(self, image: nib.Nifti1Image) -> nib.Nifti1Image:
        """Run TotalSegmentator on nibabel image."""
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            input_path = Path(tmp.name)
        nib.save(image, input_path)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            output_path = Path(tmp.name)

        print(f"[TotalSeg] Input from nib.Nifti1Image, shape={image.shape}")
        self._run_subprocess(input_path, output_path)

        input_path.unlink()
        output_img = nib.load(output_path)
        output_path.unlink()

        print(f"[TotalSeg] Done, shape={output_img.shape}")
        return output_img

    # ── run_nib: alias ──

    def run_nib(self, image: nib.Nifti1Image) -> nib.Nifti1Image:
        """Alias for run()."""
        return self.run(image)
