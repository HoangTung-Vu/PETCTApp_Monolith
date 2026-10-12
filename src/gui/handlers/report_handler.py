"""Report handler mixin for MainWindow."""

from PyQt6.QtWidgets import QMessageBox, QFileDialog


class ReportHandlerMixin:
    """Handles lesion statistics display (table, on-canvas boxes, lesion IDs)
    and report export.

    The statistics live in ``SessionManager.lesion_stats``; they are computed
    by the load / save workers, never here.
    """

    # ── Lesion statistics display ──

    def _refresh_lesion_ui(self):
        """Show the session's current lesion statistics everywhere."""
        sm = self.session_manager
        self._refresh_lesion_table()
        shape = sm.pet_image.shape if sm.pet_image is not None else None
        self.layout_manager.set_lesions(sm.lesion_stats, shape, sm.lesion_stats_stale)
        if self.control_panel.chk_show_lesion_ids.isChecked():
            self._show_lesion_ids()

    def _refresh_lesion_table(self):
        sm = self.session_manager
        message = ""
        if sm.lesion_stats is None:
            if sm.tumor_mask is None:
                message = "No segmentation loaded"
            elif sm.pet_image is None:
                message = "Load a PET image to compute lesion values"
            else:
                message = "Lesion values unavailable (see Logs)"
        self.control_panel.show_lesion_stats(sm.lesion_stats, sm.lesion_stats_stale, message)

    def _mark_lesion_stats_stale(self):
        """The tumor mask was edited: keep the values, flag them as outdated."""
        sm = self.session_manager
        sm.invalidate_lesion_stats()
        if sm.lesion_stats_stale:
            self._refresh_lesion_table()
            self.layout_manager.set_lesion_stale(True)

    def _on_lesion_double_clicked(self, pos_zyx: list):
        """Toggle the quantification box of the lesion under a double-click."""
        if self.control_panel.eraser_tab.btn_eraser_toggle.isChecked():
            return      # double-click erases the lesion in eraser mode
        if self.layout_manager._ruler_enabled:
            return      # double-click clears the ruler
        if self.current_tool != "pan_zoom" or self._manual_edit_tool != "pan_zoom":
            return      # a Segmentation-tab drawing tool is active
        result = self.layout_manager.toggle_lesion_info(pos_zyx)
        if result is None:
            if self.session_manager.lesion_stats is None:
                print("[Lesion] No lesion values for this session.")
            return
        lesion_id, shown = result
        print(f"[Lesion] Lesion {lesion_id} info {'shown' if shown else 'hidden'}.")

    def _show_lesion_ids(self):
        stats = self.session_manager.lesion_stats
        lesions = stats["lesions"] if stats else []
        if lesions:
            self.layout_manager.show_lesion_ids(
                [les["bbox"] for les in lesions], [les["id"] for les in lesions]
            )
        else:
            self.layout_manager.hide_lesion_ids()
        return bool(lesions)

    def _on_toggle_lesion_ids(self, checked: bool):
        """Show or hide lesion ID labels on all viewers."""
        if checked:
            if not self._show_lesion_ids():
                print("[Report] No lesions to label.")
        else:
            self.layout_manager.hide_lesion_ids()

    # ── Report export ──

    def _on_report_clicked(self):
        """Export the session's lesion values (CSV + images) to a chosen folder."""
        sm = self.session_manager
        if sm.current_session_id is None:
            QMessageBox.warning(self, "No Session", "Please load or create a session first.")
            return
        stats = sm.lesion_stats
        if stats is None:
            QMessageBox.warning(
                self, "No Lesion Values",
                "Load a PET image and a segmentation to export a report.",
            )
            return
        if sm.lesion_stats_stale:
            reply = QMessageBox.question(
                self, "Unsaved Edits",
                "The segmentation has edits that are not saved yet.\n\n"
                "Export the values of the last saved segmentation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        report_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Report Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not report_dir:
            return  # user cancelled

        from pathlib import Path
        from ..workers import ReportWorker

        # Current display params + the viewers' pre-converted ZYX volumes
        lm = self.layout_manager
        self._report_export_dir = Path(report_dir) / str(sm.current_session_id)
        self.report_worker = ReportWorker(
            stats,
            report_dir=self._report_export_dir,
            ct_zyx=lm._cached_data_zyx.get("ct"),
            pet_zyx=lm._cached_data_zyx.get("pet"),
            affine=sm.pet_image.affine,
            ct_wl=lm._ct_wl,
            pet_wl=lm._pet_wl,
            ct_colormap=lm._ct_colormap,
            pet_colormap=lm._pet_colormap,
            mask_opacity=lm._tumor_opacity,
        )
        self._spawn_worker(self.report_worker, self._on_report_finished, self._on_report_error, "report")

    def _on_report_finished(self, stats: dict):
        self._set_ui_busy(False)
        self.control_panel.hide_report_progress()
        n_lesions = len(stats.get("lesions", []))
        print(f"[Report] Exported: tTLG={stats.get('tTLG', 0.0):.2f}, {n_lesions} lesions")
        QMessageBox.information(self, "Report Exported", f"Report exported to:\n{self._report_export_dir}")

    def _on_report_error(self, error_msg: str):
        self._set_ui_busy(False)
        self.control_panel.hide_report_progress()
        print(f"[Report] Error: {error_msg}")
        QMessageBox.critical(self, "Report Failed", error_msg)
