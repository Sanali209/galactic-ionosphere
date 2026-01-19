"""
FileCardWidget - Card widget for displaying files in UExplorer.

Shows file thumbnail, name, size, and rating widget.
Only displays files (not folders).
"""
from typing import TYPE_CHECKING, Optional, List
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from loguru import logger

from src.ui.cardview.card_item_widget import CardItemWidget


class StarRatingWidget(QWidget):
    """
    Clickable 5-star rating widget.
    
    Signals:
        rating_changed(int): Emitted when user clicks a star (1-5)
    """
    rating_changed = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rating: int = 0
        self._max_stars: int = 5
        self._star_labels: List[QLabel] = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Build star widget."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        for i in range(self._max_stars):
            star = QLabel("☆")
            star.setStyleSheet("color: #ccc; font-size: 14px;")
            star.setCursor(Qt.CursorShape.PointingHandCursor)
            star.mousePressEvent = lambda e, idx=i+1: self._on_star_clicked(idx)
            layout.addWidget(star)
            self._star_labels.append(star)
        
        layout.addStretch()
    
    def _on_star_clicked(self, rating: int):
        """Handle star click."""
        # Toggle off if clicking same rating
        if rating == self._rating:
            self.set_rating(0)
        else:
            self.set_rating(rating)
        self.rating_changed.emit(self._rating)
    
    def set_rating(self, rating: int):
        """Set rating (0-5)."""
        self._rating = max(0, min(rating, self._max_stars))
        self._update_display()
    
    def _update_display(self):
        """Update star display."""
        for i, star in enumerate(self._star_labels):
            if i < self._rating:
                star.setText("★")
                star.setStyleSheet("color: #ffc107; font-size: 14px;")
            else:
                star.setText("☆")
                star.setStyleSheet("color: #ccc; font-size: 14px;")
    
    def rating(self) -> int:
        """Get current rating."""
        return self._rating


class FileCardWidget(CardItemWidget):
    """
    Card widget for displaying files.
    
    Features:
    - Thumbnail from ThumbnailService
    - File name (truncated)
    - File size display
    - 5-star rating widget
    
    Only shows files, not folders (folders are in DirectoryPanel).
    """
    
    rating_changed = Signal(str, int)  # file_id, new_rating
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
    
    def build_content(self):
        """Build card layout."""
        # Thumbnail container
        self._thumbnail_container = QWidget()
        self._thumbnail_container.setMinimumHeight(100)
        self._thumbnail_container.setStyleSheet("""
            background-color: #f0f0f0; 
            border-radius: 4px;
        """)
        
        thumb_layout = QVBoxLayout(self._thumbnail_container)
        thumb_layout.setContentsMargins(4, 4, 4, 4)
        
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setMinimumSize(80, 80)
        thumb_layout.addWidget(self._thumbnail_label, 1)
        
        self.layout.addWidget(self._thumbnail_container, 1)
        
        # File info container
        info_container = QWidget()
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(2)
        
        # File name
        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setWordWrap(True)
        self._name_label.setStyleSheet("""
            font-size: 11px; 
            font-weight: bold; 
            color: #333;
        """)
        self._name_label.setMaximumHeight(35)
        info_layout.addWidget(self._name_label)
        
        # File size
        self._size_label = QLabel()
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_label.setStyleSheet("font-size: 10px; color: #666;")
        info_layout.addWidget(self._size_label)
        
        # Rating stars
        rating_container = QWidget()
        rating_layout = QHBoxLayout(rating_container)
        rating_layout.setContentsMargins(0, 0, 0, 0)
        rating_layout.addStretch()
        
        self._rating_widget = StarRatingWidget()
        self._rating_widget.rating_changed.connect(self._on_rating_changed)
        rating_layout.addWidget(self._rating_widget)
        rating_layout.addStretch()
        
        info_layout.addWidget(rating_container)
        
        self.layout.addWidget(info_container)
    
    def update_display(self):
        """Update display from data context."""
        if not self._data_context:
            return
        
        # File name
        name = self._data_context.title or "Untitled"
        if len(name) > 25:
            name = name[:22] + "..."
        self._name_label.setText(name)
        
        # File size
        self._size_label.setText(self._data_context.subtitle or "")
        
        # Rating - CardItem has rating field directly
        rating = self._data_context.rating or 0
        self._rating_widget.set_rating(int(rating))
        
        # Load thumbnail
        self._load_thumbnail()
    
    def _load_thumbnail(self):
        """Load thumbnail from ThumbnailService."""
        if not self._thumbnail_service or not self._data_context:
            self._set_file_icon()
            return
        
        # Use file extension to determine icon if no thumbnail
        import asyncio
        asyncio.create_task(self._async_load_thumbnail())
    
    async def _async_load_thumbnail(self):
        """Async thumbnail loading using ThumbnailService.get_or_create()."""
        try:
            # Get file ID from data context
            file_id = self._data_context.thumbnail_path
            size = self._thumbnail_size[0] if isinstance(self._thumbnail_size, tuple) else 256
            
            from bson import ObjectId
            oid = ObjectId(file_id) if isinstance(file_id, str) else file_id
            
            # Use get_or_create which generates on-demand if not cached
            thumbnail_bytes = await self._thumbnail_service.get_or_create(oid, size)
            
            if thumbnail_bytes:
                # Create pixmap from bytes
                from PySide6.QtGui import QImage
                image = QImage()
                if image.loadFromData(thumbnail_bytes):
                    pixmap = QPixmap.fromImage(image)
                    if not pixmap.isNull():
                        # Scale to fit card
                        scaled = pixmap.scaled(
                            self._thumbnail_label.width() - 8,
                            self._thumbnail_label.height() - 8,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self._thumbnail_label.setPixmap(scaled)
                        return
            
            # Fallback to file icon
            self._set_file_icon()
            
        except Exception as e:
            logger.debug(f"Failed to load thumbnail: {e}")
            self._set_file_icon()
    
    def _set_file_icon(self):
        """Set file type icon as placeholder."""
        # Determine icon based on file type from CardItem.data or item_type
        ext = ""
        if self._data_context:
            # Try to get extension from data (FileRecord)
            if self._data_context.data and hasattr(self._data_context.data, 'extension'):
                ext = self._data_context.data.extension or ""
            elif self._data_context.item_type:
                ext = "." + self._data_context.item_type
        
        # Map extensions to emoji icons
        icon_map = {
            ".jpg": "🖼️", ".jpeg": "🖼️", ".png": "🖼️", ".gif": "🖼️",
            ".bmp": "🖼️", ".webp": "🖼️", ".svg": "🖼️",
            ".mp4": "🎬", ".avi": "🎬", ".mkv": "🎬", ".mov": "🎬",
            ".mp3": "🎵", ".wav": "🎵", ".flac": "🎵", ".aac": "🎵",
            ".pdf": "📄", ".doc": "📝", ".docx": "📝", ".txt": "📝",
            ".zip": "📦", ".rar": "📦", ".7z": "📦",
            ".py": "🐍", ".js": "📜", ".html": "🌐", ".css": "🎨",
            # Add image types for item_type
            "image": "🖼️", "video": "🎬", "audio": "🎵", "document": "📄",
        }
        
        icon = icon_map.get(ext.lower(), icon_map.get(self._data_context.item_type if self._data_context else "", "📄"))
        
        self._thumbnail_label.setText(icon)
        self._thumbnail_label.setStyleSheet("""
            font-size: 48px;
            color: #666;
        """)
    
    def _on_rating_changed(self, new_rating: int):
        """Handle rating change from widget."""
        if self._data_context:
            # Emit signal for saving
            self.rating_changed.emit(self._data_context.id, new_rating)
            logger.debug(f"Rating changed: {self._data_context.id} -> {new_rating}")
    
    def clear_content(self):
        """Clear content for recycling."""
        if hasattr(self, '_thumbnail_label'):
            self._thumbnail_label.clear()
            self._thumbnail_label.setStyleSheet("")
        if hasattr(self, '_name_label'):
            self._name_label.clear()
        if hasattr(self, '_size_label'):
            self._size_label.clear()
        if hasattr(self, '_rating_widget'):
            self._rating_widget.set_rating(0)
