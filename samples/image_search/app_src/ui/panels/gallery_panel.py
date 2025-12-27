"""
Gallery panel for displaying image search results using CardView.

Uses Foundation CardView for virtualized image display with
sort, filter, group, and selection support.
"""
from typing import List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSlider, QComboBox
from PySide6.QtCore import Signal, Qt
from loguru import logger

from src.ui.cardview.card_view import CardView
from src.ui.cardview.card_viewmodel import CardViewModel
from src.ui.cardview.models.card_item import CardItem


class GalleryPanel(QWidget):
    """
    Gallery panel displaying images using CardView.
    
    Features:
    - Virtualized card grid (handles 1000+ images)
    - Sort by title, dimensions, date
    - Group by search query
    - Multi-selection with Ctrl/Shift+click
    - Thumbnail async loading
    """
    
    selection_changed = Signal(list)  # Emits list of selected CardItems
    
    def __init__(self, title: str, viewmodel, parent: QWidget = None):
        """Initialize gallery panel."""
        super().__init__(parent)
        
        self.viewmodel = viewmodel
        self.locator = viewmodel.locator
        self._card_viewmodel: Optional[CardViewModel] = None
        self._card_view: Optional[CardView] = None
        
        self._setup_ui()
        logger.debug("GalleryPanel initialized with CardView")
    
    def _setup_ui(self):
        """Build the gallery panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # CardView with custom ImageCardWidget for URL thumbnails
        from app_src.ui.widgets.image_card_widget import ImageCardWidget
        
        self._card_viewmodel = CardViewModel(self.locator)
        self._card_view = CardView(self)
        self._card_view._default_factory = ImageCardWidget  # Use custom widget for URL images
        self._card_view.set_data_context(self._card_viewmodel)
        
        # Connect signals
        self._card_view.selection_changed.connect(self._on_selection_changed)
        self._card_view.item_double_clicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self._card_view, 1)  # 1 = stretch factor
        
        # Empty state
        self._empty_label = QLabel("No images to display\n\nPerform a search to see results")
        self._empty_label.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 14px;
                padding: 40px;
            }
        """)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.hide()
        layout.addWidget(self._empty_label)
    
    def _create_toolbar(self) -> QWidget:
        """Create toolbar with sort, group, size controls."""
        toolbar = QWidget()
        toolbar.setFixedHeight(35)
        toolbar.setStyleSheet("background-color: #f0f0f0; border-bottom: 1px solid #ccc;")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        
        # Sort dropdown
        layout.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Title", "Dimensions", "Date", "ID"])
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        layout.addWidget(self._sort_combo)
        
        # Group dropdown
        layout.addWidget(QLabel("Group:"))
        self._group_combo = QComboBox()
        self._group_combo.addItems(["None", "Search Query", "Source"])
        self._group_combo.currentTextChanged.connect(self._on_group_changed)
        layout.addWidget(self._group_combo)
        
        layout.addStretch()
        
        # Size slider
        layout.addWidget(QLabel("Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(100)
        self._size_slider.setMaximum(400)
        self._size_slider.setValue(200)
        self._size_slider.setFixedWidth(120)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        layout.addWidget(self._size_slider)
        
        return toolbar
    
    # --- Data Loading ---
    
    async def load_images(self, image_records: list):
        """
        Load images into the gallery.
        
        Args:
            image_records: List of ImageRecord ORM objects
        """
        logger.info(f"📸 Loading {len(image_records)} images into CardView gallery")
        
        if not image_records:
            self._card_view.hide()
            self._empty_label.show()
            return
        
        self._empty_label.hide()
        self._card_view.show()
        
        # Convert ImageRecord → CardItem
        items = []
        for record in image_records:
            item = self._image_record_to_card_item(record)
            items.append(item)
        
        # Load into CardViewModel
        await self._card_viewmodel.load_items(items)
        
        logger.info(f"✅ Gallery loaded with {len(items)} images")
    
    def _image_record_to_card_item(self, record) -> CardItem:
        """Convert ImageRecord to CardItem."""
        return CardItem(
            id=str(record._id),
            title=record.title[:50] if record.title else "Untitled",
            subtitle=f"{record.width}x{record.height}",
            thumbnail_path=record.thumbnail_url,
            item_type="image",
            metadata={
                "url": record.url,
                "source": getattr(record, "source", "unknown"),
                "width": record.width,
                "height": record.height,
                "search_id": str(getattr(record, "search_id", "")),
            }
        )
    
    # --- Selection ---
    
    def select_all(self):
        """Select all visible items."""
        if self._card_viewmodel:
            all_ids = [item.id for item in self._card_viewmodel.visible_items]
            self._card_viewmodel.select_items(all_ids)
    
    def clear_selection(self):
        """Clear all selections."""
        if self._card_view:
            self._card_view.clear_selection()
    
    def get_selected_items(self) -> List[CardItem]:
        """Get list of selected items."""
        if self._card_viewmodel:
            return self._card_viewmodel.get_selected_items()
        return []
    
    def get_all_items(self) -> List[CardItem]:
        """Get all items in gallery."""
        if self._card_viewmodel:
            return list(self._card_viewmodel.visible_items)
        return []
    
    # --- Event Handlers ---
    
    def _on_selection_changed(self, selected_ids: list):
        """Forward selection changes."""
        selected = self.get_selected_items()
        self.selection_changed.emit(selected)
    
    def _on_item_double_clicked(self, item_id: str):
        """Handle double-click to open image."""
        logger.info(f"Double-clicked: {item_id}")
        # TODO: Open image in viewer
    
    def _on_sort_changed(self, text: str):
        """Handle sort dropdown change."""
        field_map = {
            "Title": "title",
            "Dimensions": "subtitle",
            "Date": "id",
            "ID": "id"
        }
        field = field_map.get(text, "title")
        if self._card_viewmodel:
            self._card_viewmodel.sort_by_field(field)
    
    def _on_group_changed(self, text: str):
        """Handle group dropdown change."""
        if self._card_viewmodel:
            if text == "None":
                self._card_viewmodel.clear_grouping()
            elif text == "Search Query":
                self._card_viewmodel.group_by_field("metadata.search_id")
            elif text == "Source":
                self._card_viewmodel.group_by_field("metadata.source")
    
    def _on_size_changed(self, value: int):
        """Handle size slider change."""
        height = int(value * 1.2)  # 4:3 aspect
        if self._card_view:
            self._card_view.set_card_size(value, height)
