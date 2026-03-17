from PyQt6.QtCore import QThread, pyqtSignal

class DataLoaderWorker(QThread):
    """
    Loads NIfTI files and updates the SessionManager in a background thread 
    to prevent UI freezing.
    """
    finished = pyqtSignal(bool)  # Emits True when done
    error = pyqtSignal(str)

    def __init__(self, session_manager, current_session_id=None, ct_path=None, pet_path=None, action="update", new_doctor=None, new_patient=None):
        """
        action: "update" (existing session), "create" (new session), "load" (load existing by ID)
        """
        super().__init__()
        self.session_manager = session_manager
        self.current_session_id = current_session_id
        self.ct_path = ct_path
        self.pet_path = pet_path
        self.action = action
        self.new_doctor = new_doctor
        self.new_patient = new_patient

    def run(self):
        try:
            print(f"[DataLoaderWorker] Starting async data loading (Action: {self.action})...")
            
            if self.action == "create":
                self.session_manager.create_session(
                    self.new_doctor, 
                    self.new_patient, 
                    ct_path=self.ct_path, 
                    pet_path=self.pet_path
                )
            elif self.action == "update":
                # For update, we might need a temporary session if none exists
                if self.session_manager.current_session_id is None:
                    self.session_manager.create_session(
                        "System", 
                        "Anonymous", 
                        ct_path=self.ct_path, 
                        pet_path=self.pet_path
                    )
                else:
                    self.session_manager.update_current_session(
                        ct_path=self.ct_path, 
                        pet_path=self.pet_path
                    )
            elif self.action == "load":
                if self.current_session_id is not None:
                     self.session_manager.load_session(self.current_session_id)
                else:
                     raise ValueError("Session ID required for loading.")
                     
            # Pre-compute get_fdata() here in the background thread so it's cached in memory
            # This is the operation that usually blocks the main thread
            if self.session_manager.ct_image is not None:
                _ = self.session_manager.ct_image.get_fdata()
            if self.session_manager.pet_image is not None:
                _ = self.session_manager.pet_image.get_fdata()

            self.finished.emit(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
