from PyQt6.QtCore import QThread, pyqtSignal


class ReportWorker(QThread):
    """Computes the clinical PET report and exports CSV + images in a background thread."""

    finished = pyqtSignal(dict)   # metrics dict
    error = pyqtSignal(str)

    def __init__(self, session_manager, ct_wl=None, pet_wl=None, ct_colormap="gray", pet_colormap="jet", mask_opacity=0.7):
        super().__init__()
        self.session_manager = session_manager
        self.ct_wl = ct_wl or (350.0, 35.0)
        self.pet_wl = pet_wl or (10.0, 5.0)
        self.ct_colormap = ct_colormap
        self.pet_colormap = pet_colormap
        self.mask_opacity = mask_opacity

    def run(self):
        try:
            metrics = self.session_manager.generate_report()

            # Export CSV + per-tumor images
            self._export(metrics)

            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))

    def _export(self, metrics):
        from ...core.engine.report_engine import ReportEngine
        from ...core.file_manager import FileManager
        import numpy as np

        sm = self.session_manager
        session_id = sm.current_session_id
        if session_id is None:
            return

        report_dir = FileManager.get_session_dir(session_id) / "report"

        ct_data = sm.get_ct_data().astype(np.float32) if sm.ct_image else None
        pet_data = sm.get_pet_data().astype(np.float32) if sm.pet_image else None
        mask_data = sm.get_tumor_mask_data() if sm.tumor_mask else None

        if pet_data is None or mask_data is None:
            return

        # If no CT, create zeros
        if ct_data is None:
            ct_data = np.zeros_like(pet_data)

        affine = sm.pet_image.affine

        ReportEngine.export_report(
            report_dir=report_dir,
            metrics=metrics,
            ct_data=ct_data,
            pet_data=pet_data,
            mask_data=mask_data.astype(np.uint8),
            affine=affine,
            ct_wl=self.ct_wl,
            pet_wl=self.pet_wl,
            ct_colormap=self.ct_colormap,
            pet_colormap=self.pet_colormap,
            mask_opacity=self.mask_opacity,
        )
        print(f"[Report] Exported to {report_dir}")
