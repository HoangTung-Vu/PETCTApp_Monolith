from PyQt6.QtCore import QThread, pyqtSignal

class DataLoaderWorker(QThread):
    """
    Loads NIfTI files / DICOM series and updates the SessionManager in a
    background thread to prevent UI freezing.
    """
    finished = pyqtSignal(bool)  # Emits True when done
    error = pyqtSignal(str)

    def __init__(self, session_manager, current_session_id=None, ct_path=None, pet_path=None, tumor_seg_path=None, action="update", new_doctor=None, new_patient=None,
                 dicom_ct=None, dicom_pet=None, resample_mode="ct"):
        """
        action: "update" (existing session), "create" (new session), "load" (load existing by ID),
                "create_dicom" (new session from the DICOM series ``dicom_ct`` / ``dicom_pet``)
        """
        super().__init__()
        self.session_manager = session_manager
        self.current_session_id = current_session_id
        self.ct_path = ct_path
        self.pet_path = pet_path
        self.tumor_seg_path = tumor_seg_path
        self.action = action
        self.new_doctor = new_doctor
        self.new_patient = new_patient
        self.dicom_ct = dicom_ct
        self.dicom_pet = dicom_pet
        self.resample_mode = resample_mode

    def run(self):
        try:
            print(f"[DataLoaderWorker] Starting async data loading (Action: {self.action})...")

            if self.action == "create":
                self.session_manager.create_session(
                    self.new_doctor,
                    self.new_patient,
                    ct_path=self.ct_path,
                    pet_path=self.pet_path,
                    tumor_seg_path=self.tumor_seg_path
                )
            elif self.action == "create_dicom":
                self.session_manager.create_session_from_dicom(
                    self.dicom_ct,
                    self.dicom_pet,
                    self.resample_mode,
                    self.new_doctor,
                    self.new_patient,
                    log=lambda msg: print(f"[DICOM] {msg}"),
                )
            elif self.action == "update":
                # For update, we might need a temporary session if none exists
                if self.session_manager.current_session_id is None:
                    self.session_manager.create_session(
                        "System",
                        "Anonymous",
                        ct_path=self.ct_path,
                        pet_path=self.pet_path,
                        tumor_seg_path=self.tumor_seg_path
                    )
                else:
                    self.session_manager.update_current_session(
                        ct_path=self.ct_path,
                        pet_path=self.pet_path,
                        tumor_seg_path=self.tumor_seg_path
                    )
            elif self.action == "load":
                if self.current_session_id is not None:
                     self.session_manager.load_session(self.current_session_id)
                else:
                     raise ValueError("Session ID required for loading.")

            # Volumes are fully read into memory by the SessionManager loaders,
            # so nothing heavy is left for the main thread to decompress.
            self.finished.emit(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
