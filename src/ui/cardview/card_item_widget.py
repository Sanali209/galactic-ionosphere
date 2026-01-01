"""
CardItemWidget - Base widget for card items.

Provides common functionality for all card templates:
- DataContext binding
- Selection state
- Mouse event handling
- Thumbnail loading
"""
from typing import TYPE_CHECKING, Any, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtGui import QPixmap
from loguru import logger

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class CardItemWidget(QWidget):
    """
    Base widget for card items with DataContext pattern.
    
    Subclass and override build_content() for custom card templates.
    
    Features:
    - DataContext property for MVVM binding
    - Selection state management
    - Mouse event handling (click, double-click, drag)
    - Thumbnail loading with service integration
    - Reset for widget pool recycling
    
    Signals:
        clicked(item_id: str)
        double_clicked(item_id: str)
        selection_changed(selected: bool)
    
    Example:
        class ImageCardWidget(CardItemWidget):
            def build_content(self):
                self.image_label = QLabel()
                self.title_label = QLabel()
                self.layout.addWidget(self.image_label)
                self.layout.addWidget(self.title_label)
            
            def update_display(self):
                self.title_label.setText(self.data_context.title)
    """
    
    # Signals
    clicked = Signal(object)  # Emits Card Item when clicked
    double_clicked = Signal(object)  # Emits CardItem when double-clicked
    selection_changed = Signal(bool)  # Emits selection state
    find_similar_requested = Signal(str)  # Emits file_id when Find Similar is requested

    
    def __init__(self, parent: QWidget | None = None):
        """Initialize card item widget."""
        super().__init__(parent)
        self._data_context: Optional['CardItem'] = None
        self._card_view = None  # Parent CardView
        self._thumbnail_service = None
        self._selected = False
        self._thumbnail_size = (200, 200)
        self._setup_base_ui()
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def _setup_base_ui(self):
        """Setup base UI structure."""
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(4)
        
        # Build content (override in subclass)
        self.build_content()
        
        # Default style
        self._update_style()
    
    # --- DataContext Pattern ---
    
    @property
    def data_context(self) -> Optional['CardItem']:
        """Get the bound data item."""
        return self._data_context
    
    def bind_data(self, item: 'CardItem'):
        """
        Bind data to this widget.
        
        Called when widget is acquired from pool.
        
        Args:
            item: CardItem to display
        """
        self._data_context = item
        self.update_display()
    
    def reset(self):
        """
        Reset widget for recycling.
        
        Called when widget is returned to pool.
        """
        self._data_context = None
        self._selected = False
        self._update_style()
        self.clear_content()
    
    # --- Content Building (Override in Subclass) ---
    
    def build_content(self):
        """
        Build widget content.
        
        Override in subclass to create custom card layout.
        Default implementation shows title label.
        """
        self._title_label = QLabel()
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setWordWrap(True)
        self.layout.addWidget(self._title_label)
    
    def update_display(self):
        """
        Update display from data_context.
        
        Override in subclass for custom update logic.
        """
        if self._data_context:
            self._title_label.setText(self._data_context.title)
    
    def clear_content(self):
        """
        Clear content for recycling.
        
        Override in subclass if needed.
        """
        if hasattr(self, '_title_label'):
            self._title_label.clear()
    
    # --- Selection ---
    
    @property
    def selected(self) -> bool:
        """Get selection state."""
        return self._selected
    
    def set_selected(self, selected: bool):
        """
        Set selection state.
        
        Args:
            selected: Whether this item is selected
        """
        if self._selected != selected:
            self._selected = selected
            self._update_style()
            self.selection_changed.emit(selected)
    
    def _update_style(self):
        """Update visual style based on selection state."""
        if self._selected:
            self.setStyleSheet("""
                QWidget {
                    background-color: #2980b9;
                    border: 3px solid #1a5276;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #ffffff;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                QWidget:hover {
                    background-color: #f8f9fa;
                    border-color: #adb5bd;
                }
            """)
    
    # --- Size ---
    
    def set_size(self, width: int, height: int):
        """Set fixed size for this card."""
        self.setFixedSize(width, height)
    
    def sizeHint(self):
        """Return size hint."""
        return self.size()
    
    # --- Thumbnail ---
    
    def set_thumbnail_service(self, service):
        """Set thumbnail service for loading images."""
        self._thumbnail_service = service
    
    def set_thumbnail_size(self, size: tuple):
        """Set thumbnail size."""
        self._thumbnail_size = size
    
    async def load_thumbnail(self, label: QLabel):
        """
        Load thumbnail into label using service.
        
        Args:
            label: QLabel to display thumbnail in
        """
        if not self._thumbnail_service or not self._data_context:
            return
        
        if not self._data_context.thumbnail_path:
            return
        
        try:
            thumb_path = await self._thumbnail_service.get_thumbnail(
                self._data_context.id,
                self._thumbnail_size
            )
            if thumb_path:
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaled(
                        self._thumbnail_size[0],
                        self._thumbnail_size[1],
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))
        except Exception as e:
            logger.error(f"Failed to load thumbnail: {e}")
    
    # --- Mouse Events ---
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._card_view:
                self._card_view.handle_item_click(self, event)
            if self._data_context:
                self.clicked.emit(self._data_context.id)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._data_context:
                self.double_clicked.emit(self._data_context.id)
        super().mouseDoubleClickEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drag initiation."""
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self._card_view and self._card_view.get_selected_items():
                self._start_drag()
        super().mouseMoveEvent(event)
    
    def _start_drag(self):
        """Initiate drag operation with selected file IDs."""
        from PySide6.QtGui import QDrag
        from PySide6.QtCore import QMimeData, QByteArray
        
        if not self._card_view:
            return
        
        selected_items = self._card_view.get_selected_items()
        if not selected_items:
            return
        
        # Create mime data with file IDs
        mime_data = QMimeData()
        file_ids = ','.join([item.id for item in selected_items])
        mime_data.setData('application/x-file-ids', QByteArray(file_ids.encode('utf-8')))
        
        # Create drag object
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # Execute drag
        drag.exec(Qt.DropAction.CopyAction)
        
        logger.debug(f"Started drag with {len(selected_items)} file(s)")
    
    def get_group_key(self, group_param: str) -> str:
        """
        Get grouping key for this item.
        
        Args:
            group_param: Field name to group by
            
        Returns:
            Group key value
        """
        if self._data_context:
            return self._data_context.get_field(group_param) or "default"
        return "default"
    
    def _show_context_menu(self, position):
        """Show context menu for card actions."""
        if not self._data_context:
            return
        
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        menu = QMenu(self)
        
        # Find Similar Images action
        find_similar_action = QAction("🔍 Find Similar Images", self)
        find_similar_action.setToolTip("Search for visually similar images using CLIP embeddings")
        find_similar_action.triggered.connect(self._on_find_similar)
        menu.addAction(find_similar_action)
        
        # Show menu at cursor
        menu.exec(self.mapToGlobal(position))
    
    def _on_find_similar(self):
        """Handle Find Similar action."""
        if self._data_context:
            file_id = self._data_context.id  # CardItem.id contains the file ID
            logger.info(f"Find Similar requested for file: {file_id}")
            self.find_similar_requested.emit(file_id)
    
    def dispose(self):
        """Cleanup before destruction."""
        self._data_context = None
        self._card_view = None
        self._thumbnail_service = None
