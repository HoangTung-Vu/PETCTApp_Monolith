from PyQt6.QtCore import QThread, pyqtSignal


class ReportWorker(QThread):
    """Exports the session's lesion statistics (CSV + images) in a background thread.

    Nothing is computed or saved here: the statistics were computed when the
    segmentation was loaded / saved (``SessionManager.lesion_stats``).
    """

    finished = pyqtSignal(dict)   # the exported stats
    error = pyqtSignal(str)

    def __init__(self, stats, report_dir, ct_zyx, pet_zyx, affine, ct_wl=None, pet_wl=None,
                 ct_colormap="gray", pet_colormap="jet", mask_opacity=0.7):
        super().__init__()
        self.stats = stats
        self.report_dir = report_dir
        self.ct_zyx = ct_zyx
        self.pet_zyx = pet_zyx
        self.affine = affine
        self.ct_wl = ct_wl or (350.0, 35.0)
        self.pet_wl = pet_wl or (10.0, 5.0)
        self.ct_colormap = ct_colormap
        self.pet_colormap = pet_colormap
        self.mask_opacity = mask_opacity

    def run(self):
        try:
            import numpy as np
            from ...core.engine.report_engine import ReportEngine

            # No CT: render the CT images black, like before.
            ct_zyx = self.ct_zyx if self.ct_zyx is not None else np.zeros_like(self.pet_zyx)

            ReportEngine.export_report(
                report_dir=self.report_dir,
                stats=self.stats,
                ct_zyx=ct_zyx,
                pet_zyx=self.pet_zyx,
                affine=self.affine,
                ct_wl=self.ct_wl,
                pet_wl=self.pet_wl,
                ct_colormap=self.ct_colormap,
                pet_colormap=self.pet_colormap,
                mask_opacity=self.mask_opacity,
            )
            print(f"[Report] Exported to {self.report_dir}")
            self.finished.emit(self.stats)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
