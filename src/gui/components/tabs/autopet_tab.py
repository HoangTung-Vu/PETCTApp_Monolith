"""AutoPET Interactive tab: click mode, run inference, save."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QGroupBox,
    QGridLayout, QListWidget, QLabel, QProgressBar, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal


class AutoPETTab(QWidget):
    """AutoPET Interactive click mode + inference + save controls."""

    # Signals
    sig_autopet_click_mode_changed = pyqtSignal(str)  # 'tumor' or 'background'
    sig_autopet_run_clicked = pyqtSignal()
    sig_autopet_save_clicked = pyqtSignal()
    sig_autopet_clear_clicks = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Click Mode Selection
        grp_click = QGroupBox("Click Mode")
        click_layout = QGridLayout()

        self.btn_tumor_click = QPushButton("Tumor")
        self.btn_bg_click = QPushButton("Background")

        self.btn_tumor_click.setCheckable(True)
        self.btn_bg_click.setCheckable(True)
        self.btn_tumor_click.setChecked(True)

        self.autopet_click_group = QButtonGroup(self)
        self.autopet_click_group.addButton(self.btn_tumor_click)
        self.autopet_click_group.addButton(self.btn_bg_click)

        self.btn_tumor_click.clicked.connect(
            lambda: self.sig_autopet_click_mode_changed.emit("tumor")
        )
        self.btn_bg_click.clicked.connect(
            lambda: self.sig_autopet_click_mode_changed.emit("background")
        )

        click_layout.addWidget(self.btn_tumor_click, 0, 0)
        click_layout.addWidget(self.btn_bg_click, 0, 1)

        grp_click.setLayout(click_layout)
        layout.addWidget(grp_click)

        # 2. Click List
        grp_list = QGroupBox("Added Clicks")
        list_layout = QVBoxLayout()

        self.autopet_click_list = QListWidget()
        self.autopet_click_list.setMaximumHeight(120)
        list_layout.addWidget(self.autopet_click_list)

        self.btn_clear_clicks = QPushButton("Clear All Clicks")
        self.btn_clear_clicks.clicked.connect(self._on_clear_autopet_clicks)
        list_layout.addWidget(self.btn_clear_clicks)

        grp_list.setLayout(list_layout)
        layout.addWidget(grp_list)

        # 3. Info label
        info_label = QLabel("Double-click on viewer to place points.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)

        layout.addSpacing(10)

        layout.addSpacing(10)

        # 5. Run Button
        self.btn_autopet_run = QPushButton("Run AutoPET Interactive")
        self.btn_autopet_run.clicked.connect(self.sig_autopet_run_clicked.emit)
        layout.addWidget(self.btn_autopet_run)

        # 6. Save (Overwrite)
        self.btn_autopet_save = QPushButton("Save Result (Overwrite)")
        self.btn_autopet_save.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_autopet_save.clicked.connect(self.sig_autopet_save_clicked.emit)
        layout.addWidget(self.btn_autopet_save)

        # 7. Progress Bar
        self.autopet_progress = QProgressBar()
        self.autopet_progress.setRange(0, 0)  # Indeterminate
        self.autopet_progress.setVisible(False)
        layout.addWidget(self.autopet_progress)

        layout.addStretch()

    def _on_clear_autopet_clicks(self):
        self.autopet_click_list.clear()
        self.sig_autopet_clear_clicks.emit()

    def add_autopet_click_item(self, coord_zyx, label):
        """Add an entry to the click list widget."""
        text = f"[{label}] Z={coord_zyx[0]}, Y={coord_zyx[1]}, X={coord_zyx[2]}"
        self.autopet_click_list.addItem(text)

    def clear_autopet_click_list(self):
        """Clear the click list widget (called after inference)."""
        self.autopet_click_list.clear()

    def show_autopet_progress(self):
        self.autopet_progress.setVisible(True)
        self.btn_autopet_run.setEnabled(False)

    def hide_autopet_progress(self):
        self.autopet_progress.setVisible(False)
        self.btn_autopet_run.setEnabled(True)
