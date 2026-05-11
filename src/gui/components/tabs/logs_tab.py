import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal, QObject

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        self.textWritten.emit(str(text))
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
        # Save original stdout and stderr
        self.normal_stdout = sys.stdout
        self.normal_stderr = sys.stderr
        
        # Create custom streams
        self.stdout_stream = EmittingStream(self.normal_stdout)
        self.stdout_stream.textWritten.connect(self.append_log)
        
        self.stderr_stream = EmittingStream(self.normal_stderr)
        self.stderr_stream.textWritten.connect(self.append_log)
        
        # Redirect
        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_clear = QPushButton("Clear Logs")
        self.btn_clear.clicked.connect(self.clear_logs)
        toolbar.addWidget(self.btn_clear)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Log viewer
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("font-family: monospace; background-color: #1e1e1e; color: #d4d4d4;")
        layout.addWidget(self.text_edit)

    def append_log(self, text):
        self.text_edit.moveCursor(self.text_edit.textCursor().MoveOperation.End)
        self.text_edit.insertPlainText(text)
        self.text_edit.moveCursor(self.text_edit.textCursor().MoveOperation.End)

    def clear_logs(self):
        self.text_edit.clear()
