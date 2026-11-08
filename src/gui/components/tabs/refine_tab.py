"""Refine (Segmentation) tab: lesion label, manual edit (tumor), ROI tools,
SUV refinement, Adaptive/Iterative threshold computation."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFormLayout, QGroupBox,
    QSlider, QHBoxLayout, QLabel, QGridLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QProgressBar, QButtonGroup,
    QScrollArea, QToolButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..viewers.label_colors import label_qcolor, text_color_for

_LABEL_COLUMNS = 6
_LABEL_ROWS_VISIBLE = 4
_LABEL_BUTTON_H = 26


class RefineTab(QWidget):
    """Lesion label + manual edit (tumor) + ROI drawing tools + SUV refinement + threshold computation."""

    # Signals — Lesion label: every tool below edits only the selected id
    sig_label_selected = pyqtSignal(int)
    sig_give_ids_clicked = pyqtSignal()

    # Signals — Manual Edit (tumor mask)
    sig_manual_edit_tool = pyqtSignal(str)       # 'pan_zoom', 'paint', 'erase'
    sig_manual_edit_brush_changed = pyqtSignal(int)

    # Signals — ROI tools
    sig_set_tool = pyqtSignal(str)           # 'pan_zoom', 'paint', 'sphere', 'square'
    sig_brush_size_changed = pyqtSignal(int)
    sig_refine_suv_clicked = pyqtSignal(float)  # threshold
    sig_refine_adaptive_clicked = pyqtSignal(float, str, int)  # isocontour_fraction, bg_mode, border_thickness
    sig_refine_iterative_clicked = pyqtSignal(float, float, float, float, int)  # m, c1, c0, convergence_tol, max_iterations
    sig_confirm_roi_clicked = pyqtSignal()
    sig_save_refine_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_buttons = {}      # id → QToolButton (the "Another" one excluded)
        self._another_btn = None
        self._another_id = 1
        self._selected_label = 1
        self._labeled = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 0: Lesion Label — the id every tool below edits
        # ═══════════════════════════════════════════════════════════════
        layout.addWidget(self._build_label_section())
        layout.addSpacing(8)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 1: Manual Edit — paint/erase directly on tumor mask
        # ═══════════════════════════════════════════════════════════════
        grp_manual = QGroupBox("Manual Edit (Tumor Mask)")
        manual_layout = QVBoxLayout()

        # Tool buttons: Pan/Zoom, Paint, Erase
        manual_btn_layout = QHBoxLayout()
        self.btn_manual_pan = QPushButton("Pan/Zoom")
        self.btn_manual_paint = QPushButton("Paint")
        self.btn_manual_erase = QPushButton("Erase")

        self.btn_manual_pan.setCheckable(True)
        self.btn_manual_paint.setCheckable(True)
        self.btn_manual_erase.setCheckable(True)

        self.manual_tool_group = QButtonGroup(self)
        self.manual_tool_group.addButton(self.btn_manual_pan)
        self.manual_tool_group.addButton(self.btn_manual_paint)
        self.manual_tool_group.addButton(self.btn_manual_erase)
        self.btn_manual_pan.setChecked(True)

        self.btn_manual_pan.clicked.connect(lambda: self.sig_manual_edit_tool.emit("pan_zoom"))
        self.btn_manual_paint.clicked.connect(lambda: self.sig_manual_edit_tool.emit("paint"))
        self.btn_manual_erase.clicked.connect(lambda: self.sig_manual_edit_tool.emit("erase"))

        manual_btn_layout.addWidget(self.btn_manual_pan)
        manual_btn_layout.addWidget(self.btn_manual_paint)
        manual_btn_layout.addWidget(self.btn_manual_erase)
        manual_layout.addLayout(manual_btn_layout)

        # Brush size slider
        manual_brush_layout = QHBoxLayout()
        manual_brush_layout.addWidget(QLabel("Brush:"))
        self.slider_manual_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_manual_brush.setRange(1, 50)
        self.slider_manual_brush.setValue(10)
        self.lbl_manual_brush = QLabel("10")
        self.lbl_manual_brush.setFixedWidth(30)

        self.slider_manual_brush.valueChanged.connect(self.sig_manual_edit_brush_changed.emit)
        self.slider_manual_brush.valueChanged.connect(lambda v: self.lbl_manual_brush.setText(str(v)))

        manual_brush_layout.addWidget(self.slider_manual_brush)
        manual_brush_layout.addWidget(self.lbl_manual_brush)
        manual_layout.addLayout(manual_brush_layout)

        grp_manual.setLayout(manual_layout)
        layout.addWidget(grp_manual)

        layout.addSpacing(8)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 2: ROI Tools — paint ROI → threshold → merge to tumor
        # ═══════════════════════════════════════════════════════════════

        # 1. Tool Selection — 4 buttons: Pan/Zoom, Paint, Sphere, Square
        grp_tools = QGroupBox("ROI Tools")
        tools_layout = QGridLayout()

        self.btn_pan = QPushButton("Pan/Zoom")
        self.btn_paint = QPushButton("Paint")
        self.btn_sphere = QPushButton("Sphere")
        self.btn_square = QPushButton("Square")

        self.btn_pan.setCheckable(True)
        self.btn_paint.setCheckable(True)
        self.btn_sphere.setCheckable(True)
        self.btn_square.setCheckable(True)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_paint)
        self.tool_group.addButton(self.btn_sphere)
        self.tool_group.addButton(self.btn_square)

        self.btn_pan.setChecked(True)

        self.btn_confirm_roi = QPushButton("Confirm ROI")
        self.btn_confirm_roi.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.btn_confirm_roi.clicked.connect(self.sig_confirm_roi_clicked.emit)
        self.btn_confirm_roi.setEnabled(False)

        self.btn_pan.clicked.connect(lambda: self._set_tool_and_enable_confirm("pan_zoom", False))
        self.btn_paint.clicked.connect(lambda: self._set_tool_and_enable_confirm("paint", True))
        self.btn_sphere.clicked.connect(lambda: self._set_tool_and_enable_confirm("sphere", True))
        self.btn_square.clicked.connect(lambda: self._set_tool_and_enable_confirm("square", True))

        tools_layout.addWidget(self.btn_pan, 0, 0)
        tools_layout.addWidget(self.btn_paint, 0, 1)
        tools_layout.addWidget(self.btn_sphere, 1, 0)
        tools_layout.addWidget(self.btn_square, 1, 1)
        tools_layout.addWidget(self.btn_confirm_roi, 2, 0, 1, 2)

        grp_tools.setLayout(tools_layout)
        layout.addWidget(grp_tools)

        # 2. Brush Size
        grp_brush = QGroupBox("Brush Size")
        brush_layout = QHBoxLayout()
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 50)
        self.slider_brush.setValue(10)

        self.lbl_brush_size = QLabel("10")
        self.lbl_brush_size.setFixedWidth(30)

        self.slider_brush.valueChanged.connect(self.sig_brush_size_changed.emit)
        self.slider_brush.valueChanged.connect(lambda v: self.lbl_brush_size.setText(str(v)))

        brush_layout.addWidget(self.slider_brush)
        brush_layout.addWidget(self.lbl_brush_size)
        grp_brush.setLayout(brush_layout)
        layout.addWidget(grp_brush)

        # ── Dirty state indicator ──
        self.lbl_dirty = QLabel("")
        self.lbl_dirty.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        layout.addWidget(self.lbl_dirty)

        layout.addSpacing(5)

        # 3. SUV Refinement (base thresholding)
        grp_suv = QGroupBox("SUV Threshold (Base)")
        suv_layout = QFormLayout()

        self.spin_suv = QDoubleSpinBox()
        self.spin_suv.setRange(0.0, 50.0)
        self.spin_suv.setValue(2.5)
        self.spin_suv.setSingleStep(0.1)

        self.btn_refine = QPushButton("Apply: SUV Threshold")
        self.btn_refine.setToolTip("Apply SUV threshold within painted ROI.")
        self.btn_refine.clicked.connect(self._emit_refine_suv)

        suv_layout.addRow("Min SUV:", self.spin_suv)
        suv_layout.addRow(self.btn_refine)

        grp_suv.setLayout(suv_layout)
        layout.addWidget(grp_suv)

        layout.addSpacing(8)

        # 4. Adaptive Thresholding — computes threshold, shows in popup slider
        grp_adaptive = QGroupBox("Adaptive Threshold (find SUV)")
        adaptive_layout = QFormLayout()

        self.spin_isocontour = QDoubleSpinBox()
        self.spin_isocontour.setRange(0.01, 0.99)
        self.spin_isocontour.setValue(0.70)
        self.spin_isocontour.setSingleStep(0.05)
        self.spin_isocontour.setDecimals(2)

        self.combo_bg_mode = QComboBox()
        self.combo_bg_mode.addItems(["outside_isocontour", "border_pixels"])

        self.spin_border_thickness = QSpinBox()
        self.spin_border_thickness.setRange(1, 10)
        self.spin_border_thickness.setValue(3)

        self.btn_refine_adaptive = QPushButton("Compute: Adaptive Threshold")
        self.btn_refine_adaptive.setToolTip(
            "Compute optimal SUV threshold per ROI region.\n"
            "A popup slider will let you fine-tune before applying."
        )
        self.btn_refine_adaptive.clicked.connect(self._emit_refine_adaptive)

        adaptive_layout.addRow("Isocontour Fraction:", self.spin_isocontour)
        adaptive_layout.addRow("Background Mode:", self.combo_bg_mode)
        adaptive_layout.addRow("Border Thickness:", self.spin_border_thickness)
        adaptive_layout.addRow(self.btn_refine_adaptive)

        grp_adaptive.setLayout(adaptive_layout)
        layout.addWidget(grp_adaptive)

        layout.addSpacing(8)

        # 5. Iterative Thresholding — computes threshold, shows in popup slider
        grp_iterative = QGroupBox("Iterative Threshold (find SUV)")
        iterative_layout = QFormLayout()

        self.spin_m = QDoubleSpinBox()
        self.spin_m.setRange(0.1, 100.0)
        self.spin_m.setValue(7.8)
        self.spin_m.setSingleStep(0.1)

        self.spin_c1 = QDoubleSpinBox()
        self.spin_c1.setRange(0.0, 100.0)
        self.spin_c1.setValue(61.7)
        self.spin_c1.setSingleStep(1.0)

        self.spin_c0 = QDoubleSpinBox()
        self.spin_c0.setRange(0.0, 100.0)
        self.spin_c0.setValue(31.6)
        self.spin_c0.setSingleStep(1.0)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.01, 1.0)
        self.spin_tol.setValue(0.03)
        self.spin_tol.setSingleStep(0.01)
        self.spin_tol.setDecimals(3)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(1, 50)
        self.spin_iters.setValue(10)

        self.btn_refine_iterative = QPushButton("Compute: Iterative Threshold")
        self.btn_refine_iterative.setToolTip(
            "Compute optimal SUV threshold per ROI region.\n"
            "A popup slider will let you fine-tune before applying."
        )
        self.btn_refine_iterative.clicked.connect(self._emit_refine_iterative)

        iterative_layout.addRow("m (Slope):", self.spin_m)
        iterative_layout.addRow("c1 (B/S Coeff):", self.spin_c1)
        iterative_layout.addRow("c0 (Intercept):", self.spin_c0)
        iterative_layout.addRow("Tolerance:", self.spin_tol)
        iterative_layout.addRow("Max Iterations:", self.spin_iters)
        iterative_layout.addRow(self.btn_refine_iterative)

        grp_iterative.setLayout(iterative_layout)
        layout.addWidget(grp_iterative)

        layout.addSpacing(8)

        # 6. Save Refinement — explicit disk persist
        self.btn_save_refine = QPushButton("Save Refinement to Disk")
        self.btn_save_refine.setToolTip("Persist the previewed mask to disk (overwrites previous save).")
        self.btn_save_refine.clicked.connect(self.sig_save_refine_clicked.emit)
        self.btn_save_refine.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        layout.addWidget(self.btn_save_refine)

        # 7. Refinement Progress Bar
        self.refine_progress = QProgressBar()
        self.refine_progress.setRange(0, 0)  # Indeterminate
        self.refine_progress.setVisible(False)
        layout.addWidget(self.refine_progress)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # ── Lesion label section ──

    def _build_label_section(self) -> QGroupBox:
        grp = QGroupBox("Lesion Label")
        lay = QVBoxLayout()

        self._label_grid_widget = QWidget()
        self._label_grid = QGridLayout(self._label_grid_widget)
        self._label_grid.setContentsMargins(0, 0, 0, 0)
        self._label_grid.setSpacing(3)
        self._label_group = QButtonGroup(self)
        self._label_group.setExclusive(True)

        self._label_scroll = QScrollArea()
        self._label_scroll.setWidgetResizable(True)
        self._label_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._label_scroll.setWidget(self._label_grid_widget)
        lay.addWidget(self._label_scroll)

        self.lbl_label_info = QLabel("")
        self.lbl_label_info.setStyleSheet("font-weight: bold; padding: 2px;")
        lay.addWidget(self.lbl_label_info)

        self.lbl_label_hint = QLabel("")
        self.lbl_label_hint.setWordWrap(True)
        self.lbl_label_hint.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        lay.addWidget(self.lbl_label_hint)

        self.btn_give_ids = QPushButton("Give lesion ids")
        self.btn_give_ids.setToolTip(
            "Number the lesions from scratch: every non-zero voxel is tumor, each\n"
            "26-connected region becomes one lesion, ids 1..N from head to feet.\n"
            "The segmentation file is overwritten (int16) and the values recomputed."
        )
        self.btn_give_ids.clicked.connect(self.sig_give_ids_clicked.emit)
        lay.addWidget(self.btn_give_ids)

        grp.setLayout(lay)
        self.set_label_choices([], False, None, has_mask=False)
        return grp

    def _make_label_button(self, label_id: int, text: str) -> QToolButton:
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setText(text)
        btn.setFixedHeight(_LABEL_BUTTON_H)
        btn.setMinimumWidth(36)
        color = label_qcolor(label_id).name()
        btn.setStyleSheet(
            f"QToolButton {{ background: {color}; color: {text_color_for(label_id)};"
            " border: 1px solid #222; border-radius: 3px; font-weight: bold; padding: 0 4px; }"
            " QToolButton:checked { border: 3px solid #ffffff; }"
            " QToolButton:disabled { background: #555; color: #999; }"
        )
        self._label_group.addButton(btn)
        return btn

    def _clear_label_buttons(self):
        for btn in list(self._label_group.buttons()):
            self._label_group.removeButton(btn)
            self._label_grid.removeWidget(btn)
            btn.hide()                 # gone now, not when deleteLater runs
            btn.setParent(None)
            btn.deleteLater()
        self._label_buttons = {}
        self._another_btn = None

    def set_label_choices(self, ids, labeled: bool, selected=None, has_mask: bool = True):
        """Rebuild the lesion toggles and select one; emits ``sig_label_selected``.

        ``ids``: lesion ids present in the mask. A binary mask has only id 1
        (every connected region is a lesion, numbered by position). The last
        toggle, "Another", paints a new id (``max(ids) + 1``; 1 on a binary
        mask) for lesions the current ids miss — "Give lesion ids" then
        separates them. ``selected`` is kept when it is still offered.
        """
        self._clear_label_buttons()
        self._labeled = bool(labeled)
        ids = sorted(int(i) for i in ids) if labeled else ([1] if ids else [])
        self._another_id = (max(ids) + 1) if (labeled and ids) else 1

        row = col = 0
        for label_id in ids:
            btn = self._make_label_button(label_id, str(label_id))
            btn.setToolTip(f"Lesion {label_id}" if labeled
                           else "Binary mask: all lesions (id 1)")
            btn.clicked.connect(lambda _=False, v=label_id: self._select_label(v, False))
            self._label_grid.addWidget(btn, row, col)
            self._label_buttons[label_id] = btn
            col += 1
            if col == _LABEL_COLUMNS:
                row, col = row + 1, 0
        if col + 2 > _LABEL_COLUMNS:
            row, col = row + 1, 0
        self._another_btn = self._make_label_button(self._another_id, "Another")
        self._another_btn.setToolTip(
            "Draw lesions the current ids are missing"
            + (f" (new id {self._another_id})" if labeled else "")
            + ".\nPress Give lesion ids afterwards to give each one its own id."
        )
        self._another_btn.clicked.connect(lambda: self._select_label(self._another_id, True))
        self._label_grid.addWidget(self._another_btn, row, col, 1, 2)
        n_rows = row + 1
        visible_rows = min(n_rows, _LABEL_ROWS_VISIBLE)
        self._label_scroll.setFixedHeight(visible_rows * (_LABEL_BUTTON_H + 3) + 4)
        self._label_grid_widget.adjustSize()

        for btn in self._label_group.buttons():
            btn.setEnabled(has_mask)
        self.btn_give_ids.setEnabled(has_mask and bool(ids))

        if not has_mask:
            self.lbl_label_info.setText("No segmentation loaded")
            self.lbl_label_hint.setText("")
            self._selected_label = 1
            return
        if labeled:
            self.lbl_label_hint.setText(
                "The tools below edit only the selected lesion; other lesions count as "
                "background. Painting over another lesion asks whether it is the same lesion."
            )
        else:
            self.lbl_label_hint.setText(
                "Binary mask: each connected region is one lesion (ids by position, "
                "head → feet). Press Give lesion ids to edit lesions separately."
            )
        if selected is not None and int(selected) in self._label_buttons:
            self._select_label(int(selected), False)
        elif selected is not None and int(selected) == self._another_id:
            self._select_label(self._another_id, True)
        elif ids:
            self._select_label(ids[0], False)
        else:
            self._select_label(self._another_id, True)

    def _select_label(self, label_id: int, another: bool):
        btn = self._another_btn if another else self._label_buttons.get(label_id)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._selected_label = int(label_id)
        if another and self._labeled:
            self.lbl_label_info.setText(f"Editing: Another — new lesion id {label_id}")
        elif another:
            self.lbl_label_info.setText("Editing: Another lesion (binary mask)")
        elif self._labeled:
            self.lbl_label_info.setText(f"Editing: Lesion {label_id}")
        else:
            self.lbl_label_info.setText("Editing: all lesions (binary mask)")
        self.sig_label_selected.emit(self._selected_label)

    def selected_label(self) -> int:
        return self._selected_label

    # ── Signal emitters ──

    def _emit_refine_suv(self):
        val = self.spin_suv.value()
        self.sig_refine_suv_clicked.emit(val)

    def _emit_refine_adaptive(self):
        iso_frac = self.spin_isocontour.value()
        bg_mode = self.combo_bg_mode.currentText()
        border = self.spin_border_thickness.value()
        self.sig_refine_adaptive_clicked.emit(iso_frac, bg_mode, border)

    def _emit_refine_iterative(self):
        m = self.spin_m.value()
        c1 = self.spin_c1.value()
        c0 = self.spin_c0.value()
        tol = self.spin_tol.value()
        iters = self.spin_iters.value()
        self.sig_refine_iterative_clicked.emit(m, c1, c0, tol, iters)

    def _set_tool_and_enable_confirm(self, tool, enabled):
        self.btn_confirm_roi.setEnabled(enabled)
        self.sig_set_tool.emit(tool)

    # ── Public API for handler ──

    def set_refine_buttons_enabled(self, enabled: bool):
        """Enable/disable all refine/compute buttons based on is_dirty state."""
        self.btn_refine.setEnabled(enabled)
        self.btn_refine_adaptive.setEnabled(enabled)
        self.btn_refine_iterative.setEnabled(enabled)
        if enabled:
            self.lbl_dirty.setText("ROI painted — ready to refine")
            self.lbl_dirty.setStyleSheet("color: #5cb85c; font-style: italic; padding: 2px;")
        else:
            self.lbl_dirty.setText("Paint ROI first to enable refinement")
            self.lbl_dirty.setStyleSheet("color: #888; font-style: italic; padding: 2px;")

    def reset_tools(self):
        """Reset UI to reflect Pan/Zoom tool state."""
        self.btn_pan.setChecked(True)
        self.btn_confirm_roi.setEnabled(False)
        # BUG-1 FIX: Emit signal so shape drag is disabled and camera is re-enabled
        self.sig_set_tool.emit("pan_zoom")

    def reset_manual_edit(self):
        """Reset manual edit section to Pan/Zoom."""
        self.btn_manual_pan.setChecked(True)
        self.sig_manual_edit_tool.emit("pan_zoom")

    def show_refine_progress(self):
        self.refine_progress.setVisible(True)

    def hide_refine_progress(self):
        self.refine_progress.setVisible(False)
