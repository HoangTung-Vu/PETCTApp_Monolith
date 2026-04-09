"""Report handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox, QFileDialog


class ReportHandlerMixin:
    """Handles report generation, display, and lesion ID toggling."""

    def _on_report_clicked(self):
        """Spawn a worker to compute the clinical report."""
        if self.session_manager.current_session_id is None:
            QMessageBox.warning(self, "No Session", "Please load or create a session first.")
            return
        if self.session_manager.pet_image is None:
            QMessageBox.warning(self, "Missing Data", "PET image must be loaded to generate a report.")
            return

        # Ask user where to save the report
        report_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Report Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not report_dir:
            return  # user cancelled
        self._report_export_dir = report_dir

        # Auto-save session first: wait until save finishes before starting report
        from ..workers import SaveWorker
        self.report_save_worker = SaveWorker(self.session_manager)
        self.report_save_worker.finished.connect(self._start_report_worker)
        self.report_save_worker.error.connect(self._on_report_error)

        self.control_panel.show_report_progress()
        self._set_ui_busy(True)
        self.report_save_worker.start()

    def _start_report_worker(self):
        from pathlib import Path
        from ..workers import ReportWorker

        # Read current display params from layout_manager
        lm = self.layout_manager
        ct_wl = lm._ct_wl
        pet_wl = lm._pet_wl
        ct_colormap = lm._ct_colormap
        pet_colormap = lm._pet_colormap
        mask_opacity = lm._tumor_opacity

        session_id = self.session_manager.current_session_id
        report_dir = Path(self._report_export_dir) / session_id

        self.report_worker = ReportWorker(
            self.session_manager,
            report_dir=report_dir,
            ct_wl=ct_wl,
            pet_wl=pet_wl,
            ct_colormap=ct_colormap,
            pet_colormap=pet_colormap,
            mask_opacity=mask_opacity,
        )
        self.report_worker.finished.connect(self._on_report_finished)
        self.report_worker.error.connect(self._on_report_error)
        self.report_worker.start()

    def _on_report_finished(self, metrics: dict):
        self._set_ui_busy(False)
        self.control_panel.hide_report_progress()
        self.control_panel.show_report_results(metrics)

        if self.control_panel.chk_show_lesion_ids.isChecked():
            bboxes = self.session_manager.lesion_bboxes
            ids = self.session_manager.lesion_ids
            if bboxes:
                self.layout_manager.show_lesion_ids(bboxes, ids)

        n_lesions = len(metrics.get('lesions', []))
        print(f"[Report] Generated: gTLG={metrics.get('gTLG')}, {n_lesions} lesions")

    def _on_toggle_lesion_ids(self, checked: bool):
        """Show or hide lesion ID labels on all viewers."""
        if checked:
            bboxes = self.session_manager.lesion_bboxes
            ids = self.session_manager.lesion_ids
            if bboxes:
                self.layout_manager.show_lesion_ids(bboxes, ids)
            else:
                print("[Report] No lesion data available. Generate a report first.")
        else:
            self.layout_manager.hide_lesion_ids()

    def _on_report_error(self, error_msg: str):
        self._set_ui_busy(False)
        self.control_panel.hide_report_progress()
        print(f"[Report] Error: {error_msg}")
        QMessageBox.critical(self, "Report Failed", error_msg)
