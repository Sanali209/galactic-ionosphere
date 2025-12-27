"""
Progress dialog for long-running operations.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, 
                                QProgressBar, QPushButton)
from PySide6.QtCore import Qt, Signal

class ProgressDialog(QDialog):
    """
    Modal progress dialog with cancel button.
    Shows progress bar and status message.
    """
    cancelled = Signal()
    
    def __init__(self, title: str = "Progress", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 150)
        
        self._cancelled = False
        self._init_ui()
    
    def _init_ui(self):
        """Build dialog UI."""
        layout = QVBoxLayout(self)
        
        # Message label
        self.message_label = QLabel("Processing...")
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Detail label (optional)
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.detail_label)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)
    
    def set_progress(self, current: int, total: int):
        """
        Set progress value.
        
        Args:
            current: Current progress (0 to total)
            total: Total items
        """
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.detail_label.setText(f"{current} / {total}")
        else:
            self.progress_bar.setValue(0)
    
    def set_percentage(self, percentage: int):
        """Set progress percentage directly (0-100)."""
        self.progress_bar.setValue(max(0, min(100, percentage)))
    
    def set_message(self, message: str):
        """Update the main message."""
        self.message_label.setText(message)
    
    def set_detail(self, detail: str):
        """Update the detail text."""
        self.detail_label.setText(detail)
    
    def set_indeterminate(self, indeterminate: bool = True):
        """Set indeterminate mode (for unknown duration)."""
        if indeterminate:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
    
    def is_cancelled(self) -> bool:
        """Check if user cancelled the operation."""
        return self._cancelled
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.cancel_btn.setEnabled(False)
        self.set_message("Cancelling...")
        self.cancelled.emit()
