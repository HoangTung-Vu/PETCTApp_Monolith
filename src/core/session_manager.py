from typing import Optional, Dict
import nibabel as nib
import numpy as np
from pathlib import Path

from .file_manager import FileManager
from ..database.models import Session
from ..database.session_repository import SessionRepository
# We will import engines dynamically or inject them to avoid circular imports if any

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
        session = self.repository.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
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
            
        # Update DB with mask paths
        self.repository.update_mask_paths(self.current_session_id, tumor_path, organ_path)
        print(f"[SessionManager] Saved session {self.current_session_id}")

    def set_tumor_mask(self, mask_array: np.ndarray):
        """
        Updates the tumor mask in memory from a numpy array (e.g. from Napari).
        Wraps it in Nifti1Image using the CT affine.
        """
        if self.ct_image is None:
            raise ValueError("CT Image must be loaded to set mask (need affine).")
        
        # Create NIfTI image (Ensure type is uint8 or appropriate label type)
        self.tumor_mask = nib.Nifti1Image(mask_array.astype(np.uint8), self.ct_image.affine, self.ct_image.header)

    def set_organ_mask(self, mask_array: np.ndarray):
        """
        Updates the organ mask in memory.
        """
        if self.ct_image is None:
            raise ValueError("CT Image must be loaded to set mask (need affine).")
            
        self.organ_mask = nib.Nifti1Image(mask_array.astype(np.uint8), self.ct_image.affine, self.ct_image.header)

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

    def get_all_sessions(self):
        """Returns all sessions ordered by creation time."""
        return self.repository.get_all()
