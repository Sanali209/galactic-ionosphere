"""
CardView Demo Application.

Sample app for testing CardView component using Foundation bootstrap.
"""
import asyncio
import random
import sys
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QLabel, QSlider, QSpinBox,
    QToolBar, QSizePolicy
)

# Add path for imports
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ui.cardview import CardView, CardItem
from src.ui.cardview.card_viewmodel import CardViewModel
from src.ui.cardview.templates import TemplateSelector, ImageCardTemplate, DocumentCardTemplate

# Foundation imports
from src.ui.main_window import MainWindow
from src.ui.mvvm import BaseViewModel
from src.core.bootstrap import run_app
from PySide6.QtCore import Signal


class CardViewDemoViewModel(BaseViewModel):
    """ViewModel for demo application."""
    
    # Required signals for MainWindow
    statusMessageChanged = Signal(str)
    
    def __init__(self, locator=None):
        super().__init__(locator)
        self.card_vm = CardViewModel(locator)


class CardViewDemoWindow(MainWindow):
    """
    Demo window for testing CardView using Foundation.
    
    Features:
    - Load test data (configurable count)
    - Sort/Filter/Group controls
    - Thumbnail size slider
    - Performance metrics
    """
    
    def __init__(self, viewmodel: CardViewDemoViewModel):
        super().__init__(viewmodel)
        self.setWindowTitle("CardView Demo - Foundation")
        self.resize(1400, 900)
        
        self.card_vm = viewmodel.card_vm
        self._setup_demo_ui()
    
    def _setup_demo_ui(self):
        """Build the demo UI."""
        # Create central widget
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar area
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(12)
        
        # Item count
        toolbar_layout.addWidget(QLabel("Items:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(100, 100000)
        self.count_spin.setValue(1000)
        self.count_spin.setSingleStep(1000)
        toolbar_layout.addWidget(self.count_spin)
        
        # Load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self._on_load)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 4px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        toolbar_layout.addWidget(self.load_btn)
        
        toolbar_layout.addWidget(self._separator())
        
        # Search
        toolbar_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter...")
        self.search_edit.setFixedWidth(150)
        self.search_edit.textChanged.connect(self._on_search)
        toolbar_layout.addWidget(self.search_edit)
        
        toolbar_layout.addWidget(self._separator())
        
        # Sort
        toolbar_layout.addWidget(QLabel("Sort:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Title", "Type", "Rating", "ID"])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        toolbar_layout.addWidget(self.sort_combo)
        
        toolbar_layout.addWidget(self._separator())
        
        # Group
        toolbar_layout.addWidget(QLabel("Group:"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(["None", "Type", "Rating"])
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        toolbar_layout.addWidget(self.group_combo)
        
        toolbar_layout.addWidget(self._separator())
        
        # Thumbnail size
        toolbar_layout.addWidget(QLabel("Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(100, 350)
        self.size_slider.setValue(200)
        self.size_slider.setFixedWidth(100)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        toolbar_layout.addWidget(self.size_slider)
        
        self.size_label = QLabel("200")
        toolbar_layout.addWidget(self.size_label)
        
        toolbar_layout.addStretch()
        
        # Status
        self.status_label = QLabel("Ready")
        toolbar_layout.addWidget(self.status_label)
        
        layout.addWidget(toolbar_widget)
        
        # CardView
        self.card_view = CardView()
        self.card_view.set_data_context(self.card_vm)
        
        # Setup template selector
        selector = TemplateSelector()
        selector.register("image", ImageCardTemplate)
        selector.register("document", DocumentCardTemplate)
        self.card_view.set_template_selector(selector)
        
        # Connect signals
        self.card_vm.visibleItemsChanged.connect(self._on_items_changed)
        
        layout.addWidget(self.card_view)
        
        self.setCentralWidget(central)
    
    def _separator(self) -> QWidget:
        """Create a visual separator."""
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #dee2e6;")
        sep.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        return sep
    
    def _on_load(self):
        """Load test data."""
        count = self.count_spin.value()
        self.status_label.setText(f"Generating {count} items...")
        
        # Use QTimer to let UI update
        QTimer.singleShot(10, lambda: self._load_data(count))
    
    def _load_data(self, count: int):
        """Generate and load test data."""
        items = self._generate_test_items(count)
        
        # Use sync method - no async needed for demo
        self.card_vm.set_items(items)
        
        self.status_label.setText(f"Loaded {count} items")
    
    def _generate_test_items(self, count: int) -> List[CardItem]:
        """Generate test items."""
        types = ["image", "document", "video", "audio"]
        items = []
        
        for i in range(count):
            item_type = random.choice(types)
            rating = random.randint(0, 5)
            
            if item_type == "image":
                title = f"photo_{i:05d}.jpg"
                subtitle = f"{random.randint(1, 20)}MB"
            elif item_type == "document":
                exts = [".pdf", ".docx", ".xlsx", ".txt"]
                title = f"document_{i:05d}{random.choice(exts)}"
                subtitle = f"{random.randint(10, 500)}KB"
            else:
                title = f"file_{i:05d}"
                subtitle = f"{random.randint(1, 100)}MB"
            
            items.append(CardItem(
                id=str(i),
                title=title,
                subtitle=subtitle,
                item_type=item_type,
                rating=rating,
                group_key=item_type,
                tags=[item_type, f"rating_{rating}"]
            ))
        
        return items
    
    def _on_search(self, text: str):
        """Handle search text change."""
        self.card_vm.filter_by_text(text)
    
    def _on_sort_changed(self, field: str):
        """Handle sort change."""
        field_map = {
            "Title": "title",
            "Type": "item_type",
            "Rating": "rating",
            "ID": "id"
        }
        self.card_vm.sort_by_field(field_map.get(field, "title"))
    
    def _on_group_changed(self, field: str):
        """Handle group change."""
        if field == "None":
            self.card_vm.clear_grouping()
        else:
            field_map = {"Type": "item_type", "Rating": "rating"}
            self.card_vm.group_by_field(field_map.get(field, "item_type"))
    
    def _on_size_changed(self, size: int):
        """Handle thumbnail size change."""
        self.size_label.setText(str(size))
        self.card_view.set_card_size(size, int(size * 1.2))
    
    def _on_items_changed(self, items):
        """Update status on items change."""
        self.status_label.setText(f"Showing {len(items)} items")


if __name__ == "__main__":
    run_app(CardViewDemoWindow, CardViewDemoViewModel, app_name="CardView Demo")
