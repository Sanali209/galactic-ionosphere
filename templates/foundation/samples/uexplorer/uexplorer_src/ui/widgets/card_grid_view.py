"""
UExplorer - Card Grid View

A grid-based view for displaying files as cards with thumbnails.
Alternative to tree/list views for visual browsing.
"""
from typing import List, Optional
from bson import ObjectId
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QVBoxLayout, QLabel, 
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from loguru import logger


class FileCard(QFrame):
    """
    Individual file card widget.
    
    Displays thumbnail + filename + optional metadata.
    """
    
    clicked = Signal(object)  # file_id
    double_clicked = Signal(object)
    
    def __init__(
        self, 
        file_id: ObjectId,
        name: str,
        thumbnail: Optional[bytes] = None,
        rating: int = 0,
        parent=None
    ):
        super().__init__(parent)
        self.file_id = file_id
        self._selected = False
        
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(1)
        self.setFixedSize(160, 180)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedSize(150, 130)
        self.thumb_label.setStyleSheet(
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        
        if thumbnail:
            self.set_thumbnail(thumbnail)
        
        layout.addWidget(self.thumb_label)
        
        # Filename
        self.name_label = QLabel(name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(32)
        self.name_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.name_label)
        
        # Rating (optional)
        if rating > 0:
            stars = "★" * rating + "☆" * (5 - rating)
            self.rating_label = QLabel(stars)
            self.rating_label.setAlignment(Qt.AlignCenter)
            self.rating_label.setStyleSheet("color: gold; font-size: 10px;")
            layout.addWidget(self.rating_label)
        
        self._update_style()
    
    def set_thumbnail(self, data: bytes):
        """Set thumbnail from bytes."""
        try:
            image = QImage.fromData(data)
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                scaled = pixmap.scaled(
                    150, 130,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.thumb_label.setPixmap(scaled)
        except Exception as e:
            logger.debug(f"Failed to load thumbnail: {e}")
    
    @property
    def selected(self) -> bool:
        return self._selected
    
    @selected.setter
    def selected(self, value: bool):
        self._selected = value
        self._update_style()
    
    def _update_style(self):
        """Update card style based on selection state."""
        if self._selected:
            self.setStyleSheet("""
                FileCard {
                    background-color: #3a5a8a;
                    border: 2px solid #5a8aca;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                FileCard {
                    background-color: #3a3a3a;
                    border: 1px solid #4a4a4a;
                    border-radius: 6px;
                }
                FileCard:hover {
                    background-color: #4a4a4a;
                    border: 1px solid #6a6a6a;
                }
            """)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.file_id)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.file_id)
        super().mouseDoubleClickEvent(event)


class CardGridView(QScrollArea):
    """
    Grid view displaying files as cards.
    
    Features:
    - Responsive grid layout
    - Thumbnail display
    - Selection highlighting
    - Rating display
    - Scroll-based pagination for large collections
    - Context menu with "Find Similar" action
    
    Usage:
        grid = CardGridView()
        grid.set_files(file_list)
        grid.card_clicked.connect(on_file_selected)
        grid.load_more_requested.connect(on_load_more)
        grid.find_similar_requested.connect(on_find_similar)
    """
    
    card_clicked = Signal(object)  # file_id
    card_double_clicked = Signal(object)
    selection_changed = Signal(list)  # list of file_ids
    load_more_requested = Signal()  # Request more data (pagination)
    find_similar_requested = Signal(object)  # file_id for image→vector search
    
    CARD_SIZE = QSize(160, 180)
    SPACING = 10
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Container widget
        self.container = QWidget()
        self.setWidget(self.container)
        
        self.grid_layout = QGridLayout(self.container)
        self.grid_layout.setSpacing(self.SPACING)
        self.grid_layout.setContentsMargins(self.SPACING, self.SPACING, self.SPACING, self.SPACING)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self._cards: List[FileCard] = []
        self._selected_ids: set = set()
        self._thumbnail_service = None
        self._context_file_id = None  # Track right-clicked file
        
        # Pagination
        self._is_loading = False
        self._has_more = True
        
        # Connect scroll for infinite loading
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        logger.info("CardGridView initialized")
    
    def _show_context_menu(self, pos):
        """Show context menu with Find Similar action."""
        from PySide6.QtWidgets import QMenu, QAction
        
        # Find which card was right-clicked
        global_pos = self.mapToGlobal(pos)
        widget = self.container.childAt(self.container.mapFromGlobal(global_pos))
        
        # Walk up to find FileCard
        while widget and not isinstance(widget, FileCard):
            widget = widget.parent()
        
        if not widget or not isinstance(widget, FileCard):
            return
        
        self._context_file_id = widget.file_id
        
        menu = QMenu(self)
        
        find_similar = QAction("🎯 Find Similar", self)
        find_similar.triggered.connect(self._on_find_similar)
        menu.addAction(find_similar)
        
        menu.addSeparator()
        
        open_action = QAction("📂 Open", self)
        open_action.triggered.connect(lambda: self.card_double_clicked.emit(self._context_file_id))
        menu.addAction(open_action)
        
        menu.exec_(global_pos)
    
    def _on_find_similar(self):
        """Handle Find Similar action."""
        if self._context_file_id:
            logger.info(f"Find Similar requested for: {self._context_file_id}")
            self.find_similar_requested.emit(self._context_file_id)
    
    def set_thumbnail_service(self, service):
        """Set thumbnail service for loading images."""
        self._thumbnail_service = service
    
    def set_files(self, files: list):
        """
        Set files to display.
        
        Args:
            files: List of FileRecord-like objects with _id, name, rating
        """
        self._files = files  # Store for re-sorting
        self._display_files(files)
    
    def _display_files(self, files: list):
        """Internal method to display files in grid."""
        self.clear()
        
        if not files:
            return
        
        # Calculate columns based on width
        cols = max(1, (self.width() - 2 * self.SPACING) // (self.CARD_SIZE.width() + self.SPACING))
        
        for i, file in enumerate(files):
            row = i // cols
            col = i % cols
            
            # Get thumbnail if already available
            thumbnail = None
            if hasattr(file, '_thumbnail_data'):
                thumbnail = file._thumbnail_data
            
            card = FileCard(
                file_id=file._id,
                name=file.name,
                thumbnail=thumbnail,
                rating=getattr(file, 'rating', 0)
            )
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            
            self.grid_layout.addWidget(card, row, col)
            self._cards.append(card)
            
            # Load thumbnail asynchronously if not already loaded
            if not thumbnail and hasattr(self, '_thumbnail_service') and self._thumbnail_service:
                self._load_thumbnail_async(card, file._id)
        
        logger.debug(f"CardGridView: displayed {len(files)} files in {cols} columns")
    
    def _load_thumbnail_async(self, card, file_id):
        """Load thumbnail asynchronously for a card."""
        import asyncio
        
        async def load():
            try:
                thumb_bytes = await self._thumbnail_service.get_or_create(file_id, 256)
                if thumb_bytes and card:
                    card.set_thumbnail(thumb_bytes)
            except Exception as e:
                logger.debug(f"Failed to load thumbnail for {file_id}: {e}")
        
        asyncio.ensure_future(load())
    
    def set_sort(self, field: str, ascending: bool = True):
        """
        Sort currently displayed files.
        
        Args:
            field: Field to sort by (name, rating, etc.)
            ascending: True for ascending, False for descending
        """
        if not hasattr(self, '_files') or not self._files:
            return
        
        def get_sort_key(record):
            if field == "name":
                return getattr(record, 'name', '').lower()
            elif field == "rating":
                return getattr(record, 'rating', 0)
            elif field == "size":
                return getattr(record, 'size_bytes', 0)
            elif field == "modified" or field == "modified_at":
                return getattr(record, 'modified_at', None) or getattr(record, 'created_at', None)
            else:
                return getattr(record, 'name', '').lower()
        
        self._files = sorted(self._files, key=get_sort_key, reverse=not ascending)
        self._display_files(self._files)
        
        logger.debug(f"CardGridView sorted by {field} {'asc' if ascending else 'desc'}")
    
    def set_group(self, group_by: str = None):
        """
        Group files by field.
        
        Args:
            group_by: Field to group by (file_type, rating, date) or None
        """
        self._group_by = group_by
        
        if not hasattr(self, '_files') or not self._files or not group_by:
            return
        
        def get_group_key(record):
            if group_by == "file_type":
                return (getattr(record, 'file_type', 'other'), getattr(record, 'name', '').lower())
            elif group_by == "rating":
                return (-getattr(record, 'rating', 0), getattr(record, 'name', '').lower())
            elif group_by == "date":
                dt = getattr(record, 'modified_at', None) or getattr(record, 'created_at', None)
                date_str = dt.strftime('%Y-%m-%d') if dt else '1970-01-01'
                return (date_str, getattr(record, 'name', '').lower())
            else:
                return (getattr(record, 'name', '').lower(),)
        
        self._files = sorted(self._files, key=get_group_key)
        self._display_files(self._files)
        
        logger.debug(f"CardGridView grouped by {group_by}")
    
    def clear(self):
        """Clear all cards."""
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()
        self._selected_ids.clear()
        
        # Clear layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def select(self, file_ids: List[ObjectId]):
        """Select files by IDs."""
        self._selected_ids = set(file_ids)
        
        for card in self._cards:
            card.selected = card.file_id in self._selected_ids
        
        self.selection_changed.emit(list(self._selected_ids))
    
    def get_selected_ids(self) -> List[ObjectId]:
        """Get list of selected file IDs."""
        return list(self._selected_ids)
    
    def _on_card_clicked(self, file_id: ObjectId):
        """Handle card click."""
        # Toggle or replace selection based on modifiers
        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers & Qt.ControlModifier:
            # Toggle selection
            if file_id in self._selected_ids:
                self._selected_ids.discard(file_id)
            else:
                self._selected_ids.add(file_id)
        elif modifiers & Qt.ShiftModifier:
            # Range selection (simplified - just add to selection)
            self._selected_ids.add(file_id)
        else:
            # Replace selection
            self._selected_ids = {file_id}
        
        # Update card styles
        for card in self._cards:
            card.selected = card.file_id in self._selected_ids
        
        self.card_clicked.emit(file_id)
        self.selection_changed.emit(list(self._selected_ids))
    
    def _on_card_double_clicked(self, file_id: ObjectId):
        """Handle card double-click."""
        self.card_double_clicked.emit(file_id)
    
    def resizeEvent(self, event):
        """Reflow cards on resize."""
        super().resizeEvent(event)
        
        if self._cards:
            # Recalculate layout
            cols = max(1, (self.width() - 2 * self.SPACING) // (self.CARD_SIZE.width() + self.SPACING))
            
            for i, card in enumerate(self._cards):
                row = i // cols
                col = i % cols
                self.grid_layout.addWidget(card, row, col)
    
    def _on_scroll(self, value):
        """Handle scroll for infinite loading."""
        if self._is_loading or not self._has_more:
            return
        
        scrollbar = self.verticalScrollBar()
        threshold = scrollbar.maximum() - 200  # 200px from bottom
        
        if value >= threshold and scrollbar.maximum() > 0:
            self.load_more_requested.emit()
    
    def set_loading(self, loading: bool):
        """Set loading state."""
        self._is_loading = loading
    
    def set_has_more(self, has_more: bool):
        """Set whether more data is available."""
        self._has_more = has_more
    
    def append_files(self, files: list):
        """
        Append additional files to the grid (for pagination).
        
        Args:
            files: List of FileRecord-like objects
        """
        if not files:
            return
        
        # Get current column count
        cols = max(1, (self.width() - 2 * self.SPACING) // (self.CARD_SIZE.width() + self.SPACING))
        start_idx = len(self._cards)
        
        for i, file in enumerate(files):
            idx = start_idx + i
            row = idx // cols
            col = idx % cols
            
            thumbnail = None
            if hasattr(file, '_thumbnail_data'):
                thumbnail = file._thumbnail_data
            
            card = FileCard(
                file_id=file._id,
                name=file.name,
                thumbnail=thumbnail,
                rating=getattr(file, 'rating', 0)
            )
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            
            self.grid_layout.addWidget(card, row, col)
            self._cards.append(card)
        
        logger.debug(f"CardGridView: appended {len(files)} files, total={len(self._cards)}")

