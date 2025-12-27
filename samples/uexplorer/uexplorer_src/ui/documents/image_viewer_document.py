"""
ImageViewerDocument - Full-size image viewer document.

Opens when user double-clicks an image in the file browser.
Displays the image scaled to fit the document area.
"""
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from loguru import logger


class ImageViewerDocument(QWidget):
    """
    Full-size image viewer document.
    
    Features:
    - Displays image scaled to fit window
    - Scrollable for large images at 100% zoom
    - Tracks file path and title
    """
    
    # Signals
    content_changed = Signal()
    
    def __init__(self, file_path: str, title: str = "Image", parent=None):
        """
        Initialize image viewer.
        
        Args:
            file_path: Absolute path to image file
            title: Document title (usually filename)
            parent: Parent widget
        """
        super().__init__(parent)
        self._file_path = file_path
        self._title = title
        self._pixmap: Optional[QPixmap] = None
        
        self._setup_ui()
        self._load_image()
        
        logger.debug(f"ImageViewerDocument opened: {title}")
    
    def _setup_ui(self):
        """Build viewer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scrollable area for large images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet("background-color: #1e1e1e;")
        
        # Image label
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: transparent;")
        
        scroll.setWidget(self._image_label)
        layout.addWidget(scroll)
    
    def _load_image(self):
        """Load and display the image."""
        try:
            self._pixmap = QPixmap(self._file_path)
            
            if self._pixmap.isNull():
                self._image_label.setText(f"Failed to load: {self._file_path}")
                self._image_label.setStyleSheet("color: #f44336; font-size: 14px;")
                return
            
            # Scale to fit while keeping aspect ratio
            self._update_display()
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            self._image_label.setText(f"Error: {e}")
    
    def _update_display(self):
        """Update image display scaled to widget size."""
        if not self._pixmap or self._pixmap.isNull():
            return
        
        # Scale to fit container
        container_size = self.size()
        if container_size.width() < 100:
            container_size = self._pixmap.size()
        
        scaled = self._pixmap.scaled(
            container_size.width() - 20,
            container_size.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)
    
    def resizeEvent(self, event):
        """Handle resize - rescale image."""
        super().resizeEvent(event)
        self._update_display()
    
    # --- Properties for docking/session ---
    
    @property
    def title(self) -> str:
        """Document title."""
        return self._title
    
    @property
    def file_path(self) -> str:
        """Image file path."""
        return self._file_path
