import os
from pathlib import Path

class Settings:
    def __init__(self):
        # Base directory
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        
        # Storage configuration
        # Allow overriding via environment variable for Windows installation flexibility
        self.STORAGE_DIR = Path(os.getenv("PETCT_STORAGE_DIR", self.BASE_DIR / "storage"))
        
        # Subdirectories
        self.WEIGHTS_DIR = self.STORAGE_DIR / "weights"
        self.DATA_DIR = self.STORAGE_DIR / "data"  # Sessions and nii.gz files
        
        # Create directories if they don't exist
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a specific session."""
        session_dir = self.DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

settings = Settings()
