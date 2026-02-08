import shutil
import os
from pathlib import Path
from src.database.db import SessionLocal
from src.database.models import Session
from src.core.file_manager import FileManager

def simulate_segmentation_session_2():
    session_id = 2
    
    # 1. Update Database
    db = SessionLocal()
    session = db.query(Session).filter(Session.id == session_id).first()
    
    if not session:
        print(f"Session {session_id} not found!")
        return
        
    cwd = Path.cwd()
    # Files are in temp/
    tumor_src = cwd / "temp" / "tumor.nii.gz"
    organ_src = cwd / "temp" / "body.nii.gz"
    
    if not tumor_src.exists():
        print(f"Error: {tumor_src} not found in temp/")
        return
        
    # Setup Session Dir
    session_dir = FileManager.get_session_dir(session_id)
    print(f"Session Dir: {session_dir}")
    
    # Expected names by FileManager
    tumor_dest = session_dir / "tumor_seg.nii.gz"
    organ_dest = session_dir / "organ_seg.nii.gz"
    
    shutil.copy(tumor_src, tumor_dest)
    print(f"Copied tumor -> {tumor_dest}")
    
    if organ_src.exists():
        shutil.copy(organ_src, organ_dest)
        print(f"Copied body -> {organ_dest}")
    else:
        print(f"Warning: {organ_src} not found, skipping body mask.")
        
    # Update DB with absolute paths (for reference mostly)
    session.tumor_seg_path = str(tumor_dest.absolute())
    session.organ_seg_path = str(organ_dest.absolute())
    
    db.commit()
    print(f"Updated Session {session_id} in DB.")
    db.close()

if __name__ == "__main__":
    simulate_segmentation_session_2()
