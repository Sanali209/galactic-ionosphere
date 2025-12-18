"""
Metadata Panel for UExplorer.
"""
import asyncio
from datetime import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QScrollArea, QGroupBox, QFormLayout, QTextEdit,
                               QFrame, QSizePolicy, QToolButton)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from loguru import logger
from bson import ObjectId

from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.core.fs_service import FSService
from src.ucorefs.thumbnails.service import ThumbnailService
try:
    from src.ui.widgets.tag_selector import TagSelector
except ImportError:
    try:
        from tag_selector import TagSelector
    except ImportError:
        # Fallback for relative import if loaded as package
        from .tag_selector import TagSelector

class StarRating(QWidget):
    """Simple 5-star rating widget."""
    ratingChanged = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rating = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        
        self.stars = []
        for i in range(1, 6):
            btn = QToolButton()
            btn.setText("☆")  # Empty star
            btn.setStyleSheet("QToolButton { border: none; font-size: 18px; color: #ffa500; }")
            btn.clicked.connect(lambda _, r=i: self.set_rating(r, emit=True))
            btn.setFixedSize(24, 24)
            btn.setAutoRaise(True)
            layout.addWidget(btn)
            self.stars.append(btn)
            
    def set_rating(self, rating, emit=False):
        self.rating = rating
        for i, btn in enumerate(self.stars):
            if i < rating:
                btn.setText("★")  # Filled star
            else:
                btn.setText("☆")  # Empty star
        
        if emit:
            self.ratingChanged.emit(rating)


class MetadataPanel(QWidget):
    """
    Panel for displaying and editing file metadata.
    """
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        self.fs_service = locator.get_system(FSService)
        self.thumbnail_service = locator.get_system(ThumbnailService)
        
        self.current_file = None # FileRecord
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setSpacing(15)
        
        # 1. Header (Thumbnail + Basic Info)
        self._init_header()
        
        # 2. Rating & Tags
        self._init_social()
        
        # 3. Description
        self._init_description()
        
        # 4. EXIF / Details
        self._init_details()
        
        self.layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Start disabled
        self.setEnabled(False)
        
    def _init_header(self):
        group = QGroupBox("Preview")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QVBoxLayout(group)
        
        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setMinimumHeight(200)
        self.thumb_label.setStyleSheet("background-color: #1e1e1e; border-radius: 4px; color: #888888;")
        layout.addWidget(self.thumb_label)
        
        # Info
        self.name_label = QLabel("Filename.jpg")
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.name_label)
        
        info_layout = QHBoxLayout()
        self.size_label = QLabel("0 KB")
        self.type_label = QLabel("JPG Image")
        info_layout.addWidget(self.size_label)
        info_layout.addStretch()
        info_layout.addWidget(self.type_label)
        layout.addLayout(info_layout)
        
        self.layout.addWidget(group)
        
    def _init_social(self):
        group = QGroupBox("Organization")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QFormLayout(group)
        
        self.rating_widget = StarRating()
        self.rating_widget.ratingChanged.connect(self._save_rating)
        layout.addRow("Rating:", self.rating_widget)
        
        self.tag_selector = TagSelector(self.locator)
        self.tag_selector.tags_changed.connect(self._save_tags)
        layout.addRow("Tags:", self.tag_selector)
        
        self.layout.addWidget(group)
        
    def _init_description(self):
        group = QGroupBox("Description")
        group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        layout = QVBoxLayout(group)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        self.desc_edit.setPlaceholderText("Add a description...")
        # Save on focus lost? hard in Qt without event filter.
        # Add a save button?
        # Or rely on explicit "Save" action?
        # For auto-save, we can use debounced textChanged
        self.desc_edit.textChanged.connect(self._on_desc_changed)
        
        layout.addWidget(self.desc_edit)
        self.layout.addWidget(group)
        
    def _init_details(self):
        self.details_group = QGroupBox("Details")
        self.details_group.setStyleSheet("QGroupBox { color: #cccccc; background-color: #2d2d2d; border: 1px solid #3d3d3d; border-radius: 4px; padding: 10px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.details_layout = QFormLayout(self.details_group)
        
        self.dim_label = QLabel("-")
        self.created_label = QLabel("-")
        self.modified_label = QLabel("-")
        
        self.details_layout.addRow("Dimensions:", self.dim_label)
        self.details_layout.addRow("Created:", self.created_label)
        self.details_layout.addRow("Modified:", self.modified_label)
        
        self.layout.addWidget(self.details_group)

    def set_file(self, record: FileRecord):
        """Display file metadata."""
        if not record:
            self.clear()
            return
            
        self.current_file = record
        self.setEnabled(True)
        
        # Update Header
        self.name_label.setText(record.name)
        self.size_label.setText(f"{record.size_bytes / 1024:.1f} KB")
        self.type_label.setText(record.extension.upper())
        
        # Load Thumbnail
        asyncio.ensure_future(self._load_thumb(record))
        
        # Update Rating
        self.rating_widget.set_rating(getattr(record, 'rating', 0), emit=False)
        
        # Update Tags
        self.tag_selector.set_tags(getattr(record, 'tag_ids', []))
        
        # Update Description
        self.desc_edit.blockSignals(True)
        self.desc_edit.setText(getattr(record, 'description', ""))
        self.desc_edit.blockSignals(False)
        
        # Update Details
        mod_time = getattr(record, 'modified_at', None)
        self.modified_label.setText(mod_time.strftime("%Y-%m-%d %H:%M") if mod_time else "-")
        
        # TODO: EXIF dimensions if available in kwargs or metadata dict
        
    def clear(self):
        """Reset panel."""
        self.current_file = None
        self.setEnabled(False)
        self.name_label.setText("-")
        self.thumb_label.clear()
        self.thumb_label.setText("No Selection")
        self.rating_widget.set_rating(0)
        self.desc_edit.clear()
        
    async def _load_thumb(self, record):
        try:
            data = await self.thumbnail_service.get_or_create(record._id, size=256)
            if data:
                pixmap = QPixmap()
                pixmap.loadFromData(data)
                # Scale to fit
                scaled = pixmap.scaled(self.thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
            else:
                self.thumb_label.setText("No Preview")
        except Exception:
            self.thumb_label.setText("Error")

    def _save_rating(self, rating):
        if self.current_file:
            # Optimistic update
            self.current_file.rating = rating
            asyncio.ensure_future(self._persist_change())

    def _save_tags(self, tag_ids):
        if self.current_file:
            self.current_file.tag_ids = tag_ids
            asyncio.ensure_future(self._persist_change())
            
            # Refresh tag tree in main window if accessible
            try:
                from PySide6.QtWidgets import QApplication
                main_window = QApplication.instance().main_window
                if hasattr(main_window, 'tags_tree'):
                    asyncio.ensure_future(main_window.tags_tree.refresh_tags())
            except:
                pass
            
    def _on_desc_changed(self):
        # Debounce? For now just save on every char is too much db write.
        # But we don't have a timer here. 
        # Ideally, wait for focus out.
        if self.current_file:
            self.current_file.description = self.desc_edit.toPlainText()
            # Don't persist immediately on every keystroke
            # Maybe start a timer?
            pass

    async def _persist_change(self):
        if self.current_file:
            try:
                await self.current_file.save()
                logger.info(f"Saved metadata for {self.current_file.name}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    # Explicit save for description
    def save_description(self):
         asyncio.ensure_future(self._persist_change())

