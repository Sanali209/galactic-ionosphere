"""
Image card widget for displaying images from URLs.

Loads thumbnails asynchronously from URLs with caching.
"""
from typing import Optional
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from loguru import logger

from src.ui.cardview.card_item_widget import CardItemWidget


class ImageCardWidget(CardItemWidget):
    """
    Card widget that displays thumbnail images from URLs.
    
    Features:
    - Async image loading from URLs
    - Fallback placeholder
    - Title and subtitle display
    """
    
    # Shared network manager for all instances
    _network_manager: Optional[QNetworkAccessManager] = None
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # Initialize network manager if needed
        if ImageCardWidget._network_manager is None:
            ImageCardWidget._network_manager = QNetworkAccessManager()
    
    def build_content(self):
        """Build card layout with image and labels."""
        # Image container
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        self._image_label.setMinimumHeight(120)
        self.layout.addWidget(self._image_label, 1)
        
        # Title
        self._title_label = QLabel()
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setWordWrap(True)
        self._title_label.setStyleSheet("font-size: 11px; color: #333; font-weight: bold;")
        self._title_label.setMaximumHeight(35)
        self.layout.addWidget(self._title_label)
        
        # Subtitle (dimensions)
        self._subtitle_label = QLabel()
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_label.setStyleSheet("font-size: 10px; color: #666;")
        self._subtitle_label.setMaximumHeight(20)
        self.layout.addWidget(self._subtitle_label)
    
    def update_display(self):
        """Update display from data context."""
        if not self._data_context:
            return
        
        # Update labels
        title = self._data_context.title or "Untitled"
        if len(title) > 30:
            title = title[:27] + "..."
        self._title_label.setText(title)
        self._subtitle_label.setText(self._data_context.subtitle or "")
        
        # Load thumbnail from URL
        self._load_thumbnail_from_url()
    
    def _load_thumbnail_from_url(self):
        """Load thumbnail image from URL."""
        if not self._data_context:
            return
        
        # Get thumbnail URL
        thumb_url = self._data_context.thumbnail_path
        if not thumb_url:
            self._set_placeholder()
            return
        
        # Start async request
        request = QNetworkRequest(QUrl(thumb_url))
        request.setAttribute(
            QNetworkRequest.Attribute.CacheLoadControlAttribute,
            QNetworkRequest.CacheLoadControl.PreferCache
        )
        
        reply = ImageCardWidget._network_manager.get(request)
        reply.finished.connect(lambda r=reply: self._on_thumbnail_loaded(r))
    
    def _on_thumbnail_loaded(self, reply: QNetworkReply):
        """Handle thumbnail loaded response."""
        if reply.error() != QNetworkReply.NetworkError.NoError:
            logger.debug(f"Failed to load thumbnail: {reply.errorString()}")
            self._set_placeholder()
            reply.deleteLater()
            return
        
        # Load image data
        data = reply.readAll()
        pixmap = QPixmap()
        if pixmap.loadFromData(data):
            # Scale to fit
            scaled = pixmap.scaled(
                self._image_label.width() - 8,
                self._image_label.height() - 8,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)
        else:
            self._set_placeholder()
        
        reply.deleteLater()
    
    def _set_placeholder(self):
        """Set placeholder when image fails to load."""
        self._image_label.setText("🖼️")
        self._image_label.setStyleSheet("""
            background-color: #f0f0f0;
            border-radius: 4px;
            font-size: 32px;
            color: #999;
        """)
    
    def clear_content(self):
        """Clear content for recycling."""
        if hasattr(self, '_image_label'):
            self._image_label.clear()
            self._image_label.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        if hasattr(self, '_title_label'):
            self._title_label.clear()
        if hasattr(self, '_subtitle_label'):
            self._subtitle_label.clear()
