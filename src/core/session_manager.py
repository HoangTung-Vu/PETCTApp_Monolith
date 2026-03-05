from typing import Optional, Dict
import nibabel as nib
import numpy as np
from pathlib import Path

from .file_manager import FileManager
from ..database.models import Session
from ..database.session_repository import SessionRepository
# We will import engines dynamically or inject them to avoid circular imports if any
from .engine.refinement_engine import RefinementEngine
from .engine.report_engine import ReportEngine

class SessionManager:
    """
    Manages the active session state, including in-memory images and database records.
    """
    def __init__(self):
        self.current_session_id: Optional[int] = None
        self.ct_image: Optional[nib.Nifti1Image] = None
        self.pet_image: Optional[nib.Nifti1Image] = None
        
        # Masks are stored as nib.Nifti1Image in memory.
        # If they don't exist yet, they are None.
        self.tumor_mask: Optional[nib.Nifti1Image] = None
        self.organ_mask: Optional[nib.Nifti1Image] = None
        
        # Snapshot for ROI refinement (baseline before painting)
        self._tumor_mask_snapshot: Optional[np.ndarray] = None
        
        # Session Metadata
        self.patient_name: str = ""
        self.doctor_name: str = ""
        
        # Probability volume for tumor (single-channel, shape X,Y,Z, float32)
        self.tumor_prob: Optional[np.ndarray] = None

        # Per-lesion data from the last report generation
        self.lesion_bboxes: list = []      # list[tuple] — (d0_min,d1_min,d2_min,d0_max,d1_max,d2_max)
        self.lesion_ids: list = []         # list[int] — lesion IDs

        self.repository = SessionRepository()

    def create_session(self, doctor_name: str, patient_name: str, ct_path: Optional[Path] = None, pet_path: Optional[Path] = None) -> int:
        """
        Creates a new session.
        """
        # 1. Create DB record
        session = self.repository.create(
            patient_name=patient_name, 
            doctor_name=doctor_name,
            ct_path=str(ct_path) if ct_path else None,
            pet_path=str(pet_path) if pet_path else None
        )
        self.current_session_id = session.id
        self.patient_name = patient_name
        self.doctor_name = doctor_name
        
        # 2. Copy to storage (standardized names)
        ct_dest = None
        pet_dest = None
        
        if ct_path:
            ct_dest = FileManager.copy_to_storage(ct_path, self.current_session_id, "ct")
            self.ct_image = nib.load(ct_dest)
        else:
            self.ct_image = None
            
        if pet_path:
            pet_dest = FileManager.copy_to_storage(pet_path, self.current_session_id, "pet")
            self.pet_image = nib.load(pet_dest)
        else:
            self.pet_image = None
        
        # Update DB with standardized paths
        self.repository.update(
            self.current_session_id, 
            ct_path=str(ct_dest) if ct_dest else None, 
            pet_path=str(pet_dest) if pet_dest else None
        )
        
        # Reset masks
        self.tumor_mask = None
        self.organ_mask = None
        self.tumor_prob = None
        self._tumor_mask_snapshot = None    
        self.clear_lesion_data()
        
        print(f"[SessionManager] Created session {self.current_session_id}")
        return self.current_session_id
        
    def update_current_session(self, ct_path: Optional[Path] = None, pet_path: Optional[Path] = None):
        """Updates the current session with new files."""
        if self.current_session_id is None:
             raise ValueError("No active session to update.")
             
        ct_dest = None
        pet_dest = None
        
        if ct_path:
            ct_dest = FileManager.copy_to_storage(ct_path, self.current_session_id, "ct")
            self.ct_image = nib.load(ct_dest)
            
        if pet_path:
            pet_dest = FileManager.copy_to_storage(pet_path, self.current_session_id, "pet")
            self.pet_image = nib.load(pet_dest)
            
        self.repository.update(
            self.current_session_id,
            ct_path=str(ct_dest) if ct_dest else None,
            pet_path=str(pet_dest) if pet_dest else None
        )
        print(f"[SessionManager] Updated session {self.current_session_id}")

    def load_session(self, session_id: int):
        """
        Loads an existing session from storage into RAM.
        """
        self.current_session_id = session_id
        self.clear_lesion_data()
        session = self.repository.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        self.patient_name = session.patient_name or ""
        self.doctor_name = session.doctor_name or ""
        
        # Load CT/PET
        if FileManager.file_exists(session_id, "ct"):
            self.ct_image = FileManager.load_nifti(session_id, "ct")
        if FileManager.file_exists(session_id, "pet"):
            self.pet_image = FileManager.load_nifti(session_id, "pet")
            
        # Load Masks if they exist
        if FileManager.file_exists(session_id, "tumor_seg"):
            self.tumor_mask = FileManager.load_nifti(session_id, "tumor_seg")
        else:
            self.tumor_mask = None
            
        if FileManager.file_exists(session_id, "organ_seg"):
            self.organ_mask = FileManager.load_nifti(session_id, "organ_seg")
        else:
            self.organ_mask = None
        
        # Load tumor probability volume if it exists
        if FileManager.numpy_exists(session_id, "tumor_prob"):
            self.tumor_prob = FileManager.load_numpy(session_id, "tumor_prob")
        else:
            self.tumor_prob = None

        self._tumor_mask_snapshot = None

        print(f"[SessionManager] Loaded session {session_id}")

    def save_session(self):
        """
        Persists current in-memory masks to disk and updates DB.
        """
        if self.current_session_id is None:
            print("[SessionManager] No active session to save.")
            return

        tumor_path = None
        organ_path = None

        if self.tumor_mask is not None:
            path = FileManager.save_nifti(self.tumor_mask, self.current_session_id, "tumor_seg")
            tumor_path = str(path)
            
        if self.organ_mask is not None:
            path = FileManager.save_nifti(self.organ_mask, self.current_session_id, "organ_seg")
            organ_path = str(path)
        
        # Save tumor probability volume
        if self.tumor_prob is not None:
            FileManager.save_numpy(self.tumor_prob, self.current_session_id, "tumor_prob")
            
        # Update DB with mask paths
        self.repository.update(self.current_session_id, tumor_seg_path=tumor_path, organ_seg_path=organ_path)
        
        print(f"[SessionManager] Saved session {self.current_session_id}.")

    def clear_lesion_data(self):
        """Clear cached lesion report data (bboxes and IDs)."""
        self.lesion_bboxes = []
        self.lesion_ids = []

    def set_tumor_mask(self, mask_array: np.ndarray):
        """
        Updates the tumor mask in memory from a numpy array (e.g. from Napari).
        Wraps it in Nifti1Image using the CT affine.
        """
        if self.ct_image is None:
            raise ValueError("CT Image must be loaded to set mask (need affine).")
        
        # Create NIfTI image (Ensure type is uint8 or appropriate label type)
        self.tumor_mask = nib.Nifti1Image(mask_array.astype(np.uint8), self.ct_image.affine, self.ct_image.header)
        self.clear_lesion_data()

    def set_organ_mask(self, mask_array: np.ndarray):
        """
        Updates the organ mask in memory.
        """
        if self.ct_image is None:
            raise ValueError("CT Image must be loaded to set mask (need affine).")
            
        self.organ_mask = nib.Nifti1Image(mask_array.astype(np.uint8), self.ct_image.affine, self.ct_image.header)
        self.clear_lesion_data()

    def get_ct_data(self) -> Optional[np.ndarray]:
        if self.ct_image:
            return self.ct_image.get_fdata()
        return None

    def get_pet_data(self) -> Optional[np.ndarray]:
        if self.pet_image:
            return self.pet_image.get_fdata()
        return None
        
    def get_tumor_mask_data(self) -> Optional[np.ndarray]:
        if self.tumor_mask:
            return self.tumor_mask.get_fdata()
        return None

    def get_organ_mask_data(self) -> Optional[np.ndarray]:
        if self.organ_mask:
            return self.organ_mask.get_fdata()
        return None

    def set_tumor_prob(self, prob_array: np.ndarray):
        """Set the tumor probability volume in memory."""
        self.tumor_prob = prob_array.astype(np.float32)

    def get_tumor_prob(self) -> Optional[np.ndarray]:
        """Get the tumor probability volume."""
        return self.tumor_prob

    def get_all_sessions(self):
        """Returns all sessions ordered by creation time."""
        return self.repository.get_all()

    def snapshot_current_mask(self, mask_type: str, mask_data_snapshot: Optional[np.ndarray] = None):
        """Take a snapshot of the current mask state. 
        If mask is missing, create a zeroed mask matching CT dimensions.
        """
        if mask_type == "tumor":
            if self.tumor_mask is None and self.ct_image is not None:
                print(f"[SessionManager] Creating new zeroed Tumor Mask ({self.ct_image.shape})")
                self.tumor_mask = nib.Nifti1Image(
                    np.zeros(self.ct_image.shape, dtype=np.uint8),
                    self.ct_image.affine,
                    self.ct_image.header
                )
            if self.tumor_mask:
                self._tumor_mask_snapshot = mask_data_snapshot if mask_data_snapshot is not None else self.tumor_mask.get_fdata().copy()

    def revert_to_snapshot(self, mask_type: str):
        """Discard any unsaved/unrefined changes by reverting to the snapshot."""
        if mask_type == "tumor" and self._tumor_mask_snapshot is not None:
            self.set_tumor_mask(self._tumor_mask_snapshot)
            self._tumor_mask_snapshot = None

    def get_paint_roi(self, mask_type: str) -> Optional[np.ndarray]:
        """Compute the ROI diff between current state and snapshot."""
        current_data = None
        snapshot_data = None

        if mask_type == "tumor":
            if self.tumor_mask:
                current_data = self.tumor_mask.get_fdata()
            snapshot_data = self._tumor_mask_snapshot

        if current_data is None or snapshot_data is None:
            return None

        # ROI is anywhere the mask changed (painted or erased)
        return (np.abs(current_data - snapshot_data) > 0).astype(np.uint8)

    def refine_mask(self, mask_type: str, threshold: float) -> Optional[np.ndarray]:
        """
        Refines the specified mask using SUV thresholding.
        Updates the internal mask state and returns the refined data.
        Uses the internal snapshot to scope refinement to the painted ROI.
        """
        if self.pet_image is None:
            raise ValueError("PET image required for refinement.")
            
        # Get NIfTI objects directly
        pet_img = self.pet_image
        
        if mask_type == "tumor":
            current_mask_img = self.tumor_mask
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
            
        if current_mask_img is None:
            raise ValueError(f"No {mask_type} mask available to refine.")

        # Compute ROI
        roi_mask = self.get_paint_roi(mask_type)

        # Run Refinement (returns NIfTI image)
        refined_img = RefinementEngine.refine_suv(pet_img, current_mask_img, threshold, roi_mask)
        
        # Update Internal State
        if mask_type == "tumor":
            self.tumor_mask = refined_img
            self._tumor_mask_snapshot = None # Clear snapshot after success
            
        return refined_img.get_fdata()

    def generate_report(self) -> dict:
        """Generate a clinical report by loading the tumor mask from disk.

        The mask is loaded from the persistent file
        (``storage/data/{session_id}/tumor_seg.nii.gz``) rather than the
        in-memory ``self.tumor_mask`` so that the report always reflects
        the last-saved (committed) state.

        Returns:
            dict with keys: gTLG, lesions (per-lesion SUVmax, SUVmean, MTV).
        """
        if self.current_session_id is None:
            raise ValueError("No active session.")
        if self.pet_image is None:
            raise ValueError("PET image must be loaded to generate a report.")
        if not FileManager.file_exists(self.current_session_id, "tumor_seg"):
            raise ValueError("No saved tumor mask found for this session. "
                             "Run segmentation and save first.")

        disk_mask = FileManager.load_nifti(self.current_session_id, "tumor_seg")
        result = ReportEngine.compute_report(self.pet_image, disk_mask)

        # Store lightweight lesion data in session state for GUI access
        self.lesion_bboxes = [lesion["bbox"] for lesion in result["lesions"]]
        self.lesion_ids = [lesion["id"] for lesion in result["lesions"]]

        return result
