from src.core.session_manager import SessionManager
from pathlib import Path
import os

def test_session_metadata():
    # Setup
    manager = SessionManager()
    
    # Test Creation
    doctor = "Dr. Strange"
    patient = "Peter Parker"
    print(f"Creating session for {doctor} / {patient}")
    
    session_id = manager.create_session(doctor, patient)
    print(f"Created Session ID: {session_id}")
    
    # Test Retrieval
    sessions = manager.get_all_sessions()
    print(f"Found {len(sessions)} sessions.")
    
    found = False
    for s in sessions:
        if s.id == session_id:
            found = True
            print(f"Verified Session {s.id}: Doctor={s.doctor_name}, Patient={s.patient_name}")
            assert s.doctor_name == doctor
            assert s.patient_name == patient
            break
            
    if not found:
        print("FAILED: Session not found in get_all_sessions()")
        exit(1)
        
    print("SUCCESS: Session metadata verified.")

if __name__ == "__main__":
    try:
        test_session_metadata()
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
