from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
import nibabel as nib


class SegmentationEngine(ABC):
    """Abstract base class for segmentation engines."""
    @abstractmethod
    def run(self, input_paths: Union[str, Path, List[Union[str, Path]]]) -> nib.Nifti1Image:
        """Run segmentation on input file paths.
        Args:
            input_paths: Single path or list of paths to input NIfTI files.
            
        Returns:
            Segmentation result as Nifti1Image.
        """
        pass
    
    @abstractmethod
    def run_nib(self, images: Union[nib.Nifti1Image, List[nib.Nifti1Image]]) -> nib.Nifti1Image:
        """Run segmentation on nibabel images directly.
        Args:
            images: Single or list of Nifti1Image objects.
        Returns:
            Segmentation result as Nifti1Image.
        """
        pass
    
    def save(self, result: nib.Nifti1Image, output_path: Union[str, Path]) -> None:
        """Save segmentation result to file.
        Args:
            result: Segmentation result to save.
            output_path: Path to save the result.
        """
        nib.save(result, output_path)
        print(f"[{self.__class__.__name__}] Saved to {output_path}")
