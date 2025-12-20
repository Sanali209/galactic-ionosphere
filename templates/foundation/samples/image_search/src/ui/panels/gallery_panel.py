"""
Gallery panel for displaying image search results in a grid.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from loguru import logger
import sys
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "templates/foundation"))

from src.ui.docking.panel_base import BasePanelWidget
from src.ui.widgets.grid_panel import GridPanel

# Import card widget
import importlib.util
card_path = Path(__file__).parent.parent / "widgets/thumbnail_card.py"
spec = importlib.util.spec_from_file_location("thumbnail_card", card_path)
thumbnail_card_mod = importlib.util.module_from_spec(spec)
sys.modules["thumbnail_card"] = thumbnail_card_mod
spec.loader.exec_module(thumbnail_card_mod)
ThumbnailCard = thumbnail_card_mod.ThumbnailCard


class GalleryPanel(BasePanelWidget):
    """
    Gallery panel displaying image thumbnails in a grid.
    """
    thumbnail_clicked = Signal(object)  # Emits ImageRecord
    selection_changed = Signal(list)    # Emits list of selected ImageRecords
    
    def __init__(self, title: str, locator, parent=None):
        super().__init__(title, locator, parent)
        self.image_records = []
        self.selected_records = []
        self.card_widgets = {}  # Map record -> card widget
        
        # Initialize UI components
        self.initialize_ui()
    
    def initialize_ui(self):
        """Build the gallery panel UI."""
        layout = QVBoxLayout(self._content)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Grid panel for thumbnails
        self.grid = GridPanel(columns=4, parent=self._content)
        layout.addWidget(self.grid)
        
        # Empty state label
        self.empty_label = QLabel("No images to display\n\nPerform a search to see results")
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 14px;
                padding: 40px;
            }
        """)
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)
        
        logger.debug("GalleryPanel UI initialized")
    
    def load_images(self, image_records: list):
        """
        Load images into the gallery.
        
        Args:
            image_records: List of ImageRecord objects
        """
        logger.info(f"📸 Loading {len(image_records)} images into gallery")
        
        self.image_records = image_records
        self.selected_records.clear()  # Clear selections on new load
        self.card_widgets.clear()  # Clear card references
        
        # Clear existing widgets
        self.grid.clear()
        
        if not image_records:
            self.empty_label.setVisible(True)
            self.grid.setVisible(False)
            return
        
        self.empty_label.setVisible(False)
        self.grid.setVisible(True)
        
        # Add thumbnail cards
        for idx, record in enumerate(image_records):
            try:
                card = ThumbnailCard(record)
                card.clicked.connect(self._on_thumbnail_clicked)
                self.grid.add_item(card)
                
                # Store card reference
                self.card_widgets[record._id] = card
                
                if (idx + 1) % 10 == 0:
                    logger.debug(f"  Loaded {idx + 1}/{len(image_records)} thumbnails")
            except Exception as e:
                logger.error(f"Failed to create thumbnail card: {e}")
        
        logger.info(f"✅ Gallery loaded with {len(image_records)} images")
    
    def _on_thumbnail_clicked(self, image_record):
        """Handle thumbnail click."""
        logger.debug(f"Thumbnail clicked: {image_record.title}")
        self.thumbnail_clicked.emit(image_record)
        
        # Toggle selection (compare by _id)
        selected_ids = [r._id for r in self.selected_records]
        
        if image_record._id in selected_ids:
            # Deselect
            self.selected_records = [r for r in self.selected_records if r._id != image_record._id]
            # Update visual state
            if image_record._id in self.card_widgets:
                self.card_widgets[image_record._id].set_selected(False)
                logger.debug(f"Deselected: {image_record.title}")
        else:
            # Select
            self.selected_records.append(image_record)
            # Update visual state
            if image_record._id in self.card_widgets:
                self.card_widgets[image_record._id].set_selected(True)
                logger.debug(f"✅ Selected: {image_record.title}")
        
        self.selection_changed.emit(self.selected_records)
        logger.debug(f"Total selected: {len(self.selected_records)}")
    
    def clear(self):
        """Clear all images from gallery."""
        self.grid.clear()
        self.image_records.clear()
        self.selected_records.clear()
        self.card_widgets.clear()
        self.empty_label.setVisible(True)
        self.grid.setVisible(False)
        logger.debug("Gallery cleared")
