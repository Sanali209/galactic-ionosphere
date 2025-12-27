"""
ImageCardTemplate - Card template for image items.

Shows large thumbnail with title overlay.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap

from src.ui.cardview.templates.base_template import BaseCardTemplate


class ImageCardTemplate(BaseCardTemplate):
    """
    Card template optimized for images.
    
    Features:
    - Large thumbnail area
    - Title overlay at bottom
    - Rating stars (if available)
    - Async thumbnail loading
    """
    
    def build_content(self):
        """Build image card layout."""
        # Thumbnail (large)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(150)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border-radius: 4px;
            }
        """)
        self.content_layout.addWidget(self.thumbnail_label, 1)
        
        # Title bar at bottom
        title_bar = QWidget()
        title_bar.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.6);
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)
        title_layout = QVBoxLayout(title_bar)
        title_layout.setContentsMargins(8, 4, 8, 4)
        title_layout.setSpacing(2)
        
        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_layout.addWidget(self.title_label)
        
        # Subtitle (dimensions, size)
        self.subtitle_label = QLabel()
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 10px;
            }
        """)
        title_layout.addWidget(self.subtitle_label)
        
        self.content_layout.addWidget(title_bar)
    
    def update_display(self):
        """Update from data_context."""
        if not self._data_context:
            return
        
        self.title_label.setText(self._data_context.title or "")
        self.subtitle_label.setText(self._data_context.subtitle or "")
        
        # Load thumbnail async
        if self._thumbnail_service and self._data_context.thumbnail_path:
            # Queue async load (handled by CardView)
            pass
    
    def clear_content(self):
        """Clear for recycling."""
        self.thumbnail_label.clear()
        self.title_label.clear()
        self.subtitle_label.clear()
    
    def set_thumbnail(self, pixmap: QPixmap):
        """Set thumbnail image."""
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                self.thumbnail_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)
