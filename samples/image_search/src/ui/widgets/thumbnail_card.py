"""
Thumbnail card widget for displaying image search results.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "templates/foundation"))

from src.ui.widgets.async_image import AsyncImageWidget

class ThumbnailCard(QWidget):
    """Card widget displaying image thumbnail with metadata."""
    
    clicked = Signal(object)  # Emits ImageRecord when clicked
    
    def __init__(self, image_record, parent=None):
        super().__init__(parent)
        self.image_record = image_record
        self.selected = False
        
        self.setFixedSize(200, 240)
        self.setCursor(Qt.PointingHandCursor)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Async image widget (thumbnail)
        self.image_widget = AsyncImageWidget(self)
        self.image_widget.setFixedSize(190, 190)
        
        # Load image asynchronously
        import asyncio
        asyncio.create_task(self.image_widget.load_from_url(self.image_record.thumbnail_url))
        
        layout.addWidget(self.image_widget)
        
        # Title label
        self.title_label = QLabel(self.image_record.title[:30] + "..." if len(self.image_record.title) > 30 else self.image_record.title)
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(30)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-size: 11px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # Dimensions label
        dims_text = f"{self.image_record.width}x{self.image_record.height}"
        self.dims_label = QLabel(dims_text)
        self.dims_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 10px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.dims_label)
        
        self._update_style()
    
    def _update_style(self):
        """Update widget style based on selection state."""
        if self.selected:
            self.setStyleSheet("""
                ThumbnailCard {
                    background-color: #bbdefb;
                    border: 4px solid #1976d2;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                ThumbnailCard {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                ThumbnailCard:hover {
                    border: 2px solid #2196f3;
                }
            """)
        
        # Force visual update
        self.update()
        self.repaint()
    
    def mousePressEvent(self, event):
        """Handle click event."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_record)
        super().mousePressEvent(event)
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self._update_style()
