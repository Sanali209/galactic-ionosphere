"""
CardView Module - Virtualized Card Grid Component.

A high-performance, MVVM-compatible card view with:
- Widget pool virtualization (handles 100K+ items)
- Real QWidget items with full interactivity
- Sort/Filter/Group controlled by ViewModel
- State persistence via MongoDB ORM
- ThumbnailService integration

Usage:
    from src.ui.cardview import CardView, CardViewModel, CardItem
    
    vm = CardViewModel(locator)
    card_view = CardView()
    card_view.set_data_context(vm)
    await vm.load_items(items)
"""
from src.ui.cardview.models.card_item import CardItem, SortOrder, FilterOperator
from src.ui.cardview.flow_layout import FlowLayout
from src.ui.cardview.widget_pool import WidgetPool
from src.ui.cardview.card_item_widget import CardItemWidget
from src.ui.cardview.card_group_widget import CardGroupWidget
from src.ui.cardview.card_view import CardView

__all__ = [
    # Main widget
    "CardView",
    # Item widgets
    "CardItemWidget",
    "CardGroupWidget",
    # Data types
    "CardItem",
    "SortOrder",
    "FilterOperator",
    # Layout
    "FlowLayout",
    # Virtualization
    "WidgetPool",
]
