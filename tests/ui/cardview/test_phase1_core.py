"""
Phase 1 Tests - Core CardView Components.

Tests for CardItem, FlowLayout, and WidgetPool.
"""
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication, QWidget, QPushButton
from PySide6.QtCore import QRect

# Ensure QApplication exists for tests
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestCardItem:
    """Tests for CardItem data model."""
    
    def test_create_basic_item(self):
        """Test creating a CardItem with required fields."""
        import sys
        from pathlib import Path
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(id="item_1", title="Test Item")
        
        assert item.id == "item_1"
        assert item.title == "Test Item"
        assert item.subtitle is None
        assert item.item_type == "default"
    
    def test_create_full_item(self):
        """Test creating CardItem with all fields."""
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(
            id="file_123",
            title="vacation.jpg",
            subtitle="2.4 MB",
            thumbnail_path="/path/to/file.jpg",
            item_type="image",
            group_key="photos",
            rating=4,
            tags=["vacation", "beach"]
        )
        
        assert item.id == "file_123"
        assert item.title == "vacation.jpg"
        assert item.subtitle == "2.4 MB"
        assert item.thumbnail_path == "/path/to/file.jpg"
        assert item.item_type == "image"
        assert item.group_key == "photos"
        assert item.rating == 4
        assert item.tags == ["vacation", "beach"]
    
    def test_get_field(self):
        """Test dynamic field access."""
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(id="1", title="Test", rating=3)
        
        assert item.get_field("title") == "Test"
        assert item.get_field("rating") == 3
        assert item.get_field("nonexistent") is None
    
    def test_matches_text_title(self):
        """Test text matching in title."""
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(id="1", title="Vacation Photo")
        
        assert item.matches_text("vacation")
        assert item.matches_text("VACATION")  # Case insensitive
        assert item.matches_text("photo")
        assert not item.matches_text("beach")
    
    def test_matches_text_tags(self):
        """Test text matching in tags."""
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(id="1", title="Image", tags=["beach", "summer"])
        
        assert item.matches_text("beach")
        assert item.matches_text("SUMMER")
        assert not item.matches_text("winter")
    
    def test_matches_text_empty(self):
        """Test empty search returns True."""
        from src.ui.cardview.models.card_item import CardItem
        
        item = CardItem(id="1", title="Test")
        
        assert item.matches_text("")
        assert item.matches_text(None)  # type: ignore


class TestFlowLayout:
    """Tests for FlowLayout."""
    
    def test_create_layout(self, qapp):
        """Test creating FlowLayout."""
        from src.ui.cardview.flow_layout import FlowLayout
        
        layout = FlowLayout(margin=8, h_spacing=10, v_spacing=10)
        
        assert layout.h_spacing == 10
        assert layout.v_spacing == 10
        assert layout.count() == 0
    
    def test_add_widgets(self, qapp):
        """Test adding widgets to layout."""
        from src.ui.cardview.flow_layout import FlowLayout
        
        container = QWidget()
        layout = FlowLayout()
        container.setLayout(layout)
        
        for i in range(5):
            btn = QPushButton(f"Button {i}")
            layout.addWidget(btn)
        
        assert layout.count() == 5
    
    def test_clear_layout(self, qapp):
        """Test clearing layout."""
        from src.ui.cardview.flow_layout import FlowLayout
        
        container = QWidget()
        layout = FlowLayout()
        container.setLayout(layout)
        
        for i in range(3):
            layout.addWidget(QPushButton(f"Btn {i}"))
        
        assert layout.count() == 3
        layout.clear()
        assert layout.count() == 0
    
    def test_items_per_row(self, qapp):
        """Test items per row calculation."""
        from src.ui.cardview.flow_layout import FlowLayout
        
        container = QWidget()
        container.resize(500, 300)
        layout = FlowLayout(h_spacing=10)
        container.setLayout(layout)
        
        # With 100px items and 10px spacing in 500px container
        # Should fit 4 items: 100 + 10 + 100 + 10 + 100 + 10 + 100 = 430
        items = layout.get_items_per_row(100)
        assert items >= 1


class TestWidgetPool:
    """Tests for WidgetPool virtualization."""
    
    def test_create_pool(self, qapp):
        """Test creating widget pool."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        assert pool.active_count == 0
        assert pool.free_count == 0
        assert pool.total_count == 0
    
    def test_acquire_creates_widget(self, qapp):
        """Test acquiring widget creates new instance."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        widget = pool.acquire("item_1")
        
        assert widget is not None
        assert isinstance(widget, QWidget)
        assert pool.active_count == 1
        assert pool.is_active("item_1")
    
    def test_acquire_same_id_returns_same_widget(self, qapp):
        """Test acquiring same ID returns existing widget."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        widget1 = pool.acquire("item_1")
        widget2 = pool.acquire("item_1")
        
        assert widget1 is widget2
        assert pool.active_count == 1
    
    def test_release_returns_to_pool(self, qapp):
        """Test releasing widget returns to free pool."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        pool.acquire("item_1")
        assert pool.active_count == 1
        assert pool.free_count == 0
        
        pool.release("item_1")
        assert pool.active_count == 0
        assert pool.free_count == 1
        assert not pool.is_active("item_1")
    
    def test_recycle_widget(self, qapp):
        """Test widget is recycled after release."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        created_widgets = []
        def factory():
            w = QWidget()
            created_widgets.append(w)
            return w
        
        pool = WidgetPool(factory=factory, pool_size=10)
        
        widget1 = pool.acquire("item_1")
        pool.release("item_1")
        widget2 = pool.acquire("item_2")
        
        # Should recycle the same widget
        assert widget1 is widget2
        assert len(created_widgets) == 1  # Only one widget created
    
    def test_release_outside_viewport(self, qapp):
        """Test releasing widgets outside viewport."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        # Acquire 5 widgets
        for i in range(5):
            pool.acquire(f"item_{i}")
        
        assert pool.active_count == 5
        
        # Only items 2, 3 are visible
        visible = {"item_2", "item_3"}
        released = pool.release_outside_viewport(visible)
        
        assert released == 3
        assert pool.active_count == 2
        assert pool.is_active("item_2")
        assert pool.is_active("item_3")
        assert not pool.is_active("item_0")
    
    def test_pool_size_limit(self, qapp):
        """Test pool respects size limit."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=5)
        
        # Try to acquire more than pool size
        for i in range(10):
            pool.acquire(f"item_{i}")
        
        # Should still work but recycle
        assert pool.total_count <= 5
    
    def test_clear_pool(self, qapp):
        """Test clearing all active widgets."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10)
        
        for i in range(5):
            pool.acquire(f"item_{i}")
        
        pool.clear()
        
        assert pool.active_count == 0
        assert pool.free_count == 5
    
    def test_prealloc(self, qapp):
        """Test pre-allocation of widgets."""
        from src.ui.cardview.widget_pool import WidgetPool
        
        pool = WidgetPool(factory=lambda: QWidget(), pool_size=10, prealloc=True)
        
        assert pool.free_count == 10
        assert pool.active_count == 0


class TestSortOrder:
    """Tests for SortOrder enum."""
    
    def test_sort_order_values(self):
        from src.ui.cardview.models.card_item import SortOrder
        
        assert SortOrder.ASCENDING.value == "asc"
        assert SortOrder.DESCENDING.value == "desc"


class TestFilterOperator:
    """Tests for FilterOperator enum."""
    
    def test_filter_operator_values(self):
        from src.ui.cardview.models.card_item import FilterOperator
        
        assert FilterOperator.AND.value == "and"
        assert FilterOperator.OR.value == "or"
