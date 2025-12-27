"""
CardView - Virtualized card grid widget.

Flat view: Absolute positioning with widget pool virtualization.
Grouped view: CardGroupWidgets with direct widget creation (no pool).
"""
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QFrame,
    QApplication, QSizePolicy
)
from PySide6.QtGui import QResizeEvent
from loguru import logger

from src.ui.cardview.widget_pool import WidgetPool
from src.ui.cardview.card_item_widget import CardItemWidget
from src.ui.cardview.card_group_widget import CardGroupWidget

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class CardView(QWidget):
    """
    Virtualized card grid view.
    
    Flat view: Absolute positioning with widget pool for virtualization.
    Grouped view: CardGroupWidgets with simple widget creation.
    """
    
    # Signals
    item_clicked = Signal(str)
    item_double_clicked = Signal(str)
    selection_changed = Signal(list)
    context_menu_requested = Signal(str, object)
    
    # Size presets
    SIZE_SMALL = (150, 180)
    SIZE_MEDIUM = (200, 240)
    SIZE_LARGE = (280, 320)
    
    def __init__(self, parent: QWidget | None = None):
        """Initialize CardView."""
        super().__init__(parent)
        
        # ViewModel
        self._viewmodel = None
        
        # Widget pool for flat view only
        self._widget_pool: Optional[WidgetPool[CardItemWidget]] = None
        self._pool_size = 100
        self._default_factory: Callable[[], CardItemWidget] = CardItemWidget
        
        # Settings
        self._card_width = 200
        self._card_height = 240
        self._h_spacing = 8
        self._v_spacing = 8
        self._margin = 8
        
        # Selection
        self._selected_ids: Set[str] = set()
        self._last_selected_id: Optional[str] = None
        
        # Items and visible widgets (flat view)
        self._items: List['CardItem'] = []
        self._visible_widgets: Dict[int, CardItemWidget] = {}
        self._visible_range: Tuple[int, int] = (-1, -1)
        
        # Grouping (grouped view)
        self._is_grouped = False
        self._groups: Dict[str, CardGroupWidget] = {}
        self._group_layout = None
        
        # Deferred update
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._update_visible_items)
        
        self._template_selector = None
        self._thumbnail_service = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container widget
        self.container = QWidget()
        self.container.setMouseTracking(True)
        
        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)
        
        # Connect scroll events
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)
        
        # Initialize widget pool for flat view
        self._init_widget_pool()
    
    def _init_widget_pool(self):
        """Initialize the widget pool."""
        self._widget_pool = WidgetPool(
            factory=self._create_pool_widget,
            pool_size=self._pool_size
        )
    
    def _create_pool_widget(self) -> CardItemWidget:
        """Factory for pool widgets (flat view)."""
        widget = self._default_factory(self.container)
        widget._card_view = self
        widget.set_size(self._card_width, self._card_height)
        widget.set_thumbnail_service(self._thumbnail_service)
        widget.clicked.connect(self.item_clicked.emit)
        widget.double_clicked.connect(self.item_double_clicked.emit)
        
        # Enable drag and drop
        widget.setAcceptDrops(False)  # CardView doesn't accept drops, only drags
        
        widget.hide()
        return widget
    
    def _create_group_widget(self, item: 'CardItem', parent: QWidget) -> CardItemWidget:
        """Create widget for grouped view (no pool)."""
        widget = self._default_factory(parent)
        widget._card_view = self
        widget.set_size(self._card_width, self._card_height)
        widget.set_thumbnail_service(self._thumbnail_service)
        widget.bind_data(item)
        widget.set_selected(item.id in self._selected_ids)
        widget.clicked.connect(self.item_clicked.emit)
        widget.double_clicked.connect(self.item_double_clicked.emit)
        return widget
    
    # --- Layout Calculations (Flat View) ---
    
    def _get_items_per_row(self) -> int:
        available = self.scroll_area.viewport().width() - 2 * self._margin
        return max(1, (available + self._h_spacing) // (self._card_width + self._h_spacing))
    
    def _get_total_rows(self) -> int:
        items_per_row = self._get_items_per_row()
        return (len(self._items) + items_per_row - 1) // items_per_row if items_per_row > 0 else 0
    
    def _get_total_height(self) -> int:
        total_rows = self._get_total_rows()
        return self._margin * 2 + total_rows * (self._card_height + self._v_spacing) - self._v_spacing
    
    def _get_item_position(self, index: int) -> QPoint:
        items_per_row = self._get_items_per_row()
        row = index // items_per_row
        col = index % items_per_row
        x = self._margin + col * (self._card_width + self._h_spacing)
        y = self._margin + row * (self._card_height + self._v_spacing)
        return QPoint(x, y)
    
    def _get_visible_range(self) -> Tuple[int, int]:
        scroll_y = self.scroll_area.verticalScrollBar().value()
        viewport_height = self.scroll_area.viewport().height()
        items_per_row = self._get_items_per_row()
        row_height = self._card_height + self._v_spacing
        first_row = max(0, (scroll_y - self._margin) // row_height - 1)
        last_row = (scroll_y + viewport_height + self._margin) // row_height + 2
        first_idx = first_row * items_per_row
        last_idx = min(len(self._items), last_row * items_per_row)
        return (first_idx, last_idx)
    
    # --- DataContext / ViewModel ---
    
    def set_data_context(self, viewmodel):
        """Bind ViewModel."""
        self._viewmodel = viewmodel
        if hasattr(viewmodel, 'visibleItemsChanged'):
            viewmodel.visibleItemsChanged.connect(self._on_items_changed)
        if hasattr(viewmodel, 'groupedItemsChanged'):
            viewmodel.groupedItemsChanged.connect(self._on_grouped_items_changed)
    
    @property
    def data_context(self):
        return self._viewmodel
    
    # --- Flat View ---
    
    def set_items(self, items: List['CardItem']):
        """Set items for flat view (virtualized)."""
        self._is_grouped = False
        self._items = items
        
        # Clear grouped view if exists
        self._clear_groups()
        
        # Set container height
        height = max(100, self._get_total_height())
        self.container.setFixedHeight(height)
        
        # Clear and rebuild
        self._clear_visible_widgets()
        self._visible_range = (-1, -1)
        
        self.scroll_area.verticalScrollBar().setValue(0)
        self._update_visible_items()
        
        logger.info(f"CardView: Flat view with {len(items)} items")
    
    def _on_items_changed(self, items: List['CardItem']):
        if not self._is_grouped:
            self.set_items(items)
    
    def _update_visible_items(self):
        """Update visible items for flat view."""
        if self._is_grouped or not self._items:
            return
        
        new_range = self._get_visible_range()
        if new_range == self._visible_range:
            return
        
        first_idx, last_idx = new_range
        self._visible_range = new_range
        
        # Hide out-of-range widgets
        for idx in list(self._visible_widgets.keys()):
            if idx < first_idx or idx >= last_idx:
                widget = self._visible_widgets.pop(idx)
                widget.hide()
                self._widget_pool.release(self._items[idx].id)
        
        # Show visible widgets
        for idx in range(first_idx, last_idx):
            if idx < 0 or idx >= len(self._items):
                continue
            if idx not in self._visible_widgets:
                item = self._items[idx]
                widget = self._widget_pool.acquire(item.id)
                if widget.parent() != self.container:
                    widget.setParent(self.container)
                widget.bind_data(item)
                widget.set_selected(item.id in self._selected_ids)
                pos = self._get_item_position(idx)
                widget.move(pos)
                widget.setFixedSize(self._card_width, self._card_height)
                widget.show()
                widget.raise_()
                self._visible_widgets[idx] = widget
        
        logger.debug(f"Visible: {first_idx}-{last_idx}, widgets: {len(self._visible_widgets)}")
    
    def _clear_visible_widgets(self):
        """Clear flat view widgets."""
        for idx, widget in self._visible_widgets.items():
            widget.hide()
            if idx < len(self._items):
                self._widget_pool.release(self._items[idx].id)
        self._visible_widgets.clear()
        self._widget_pool.clear()
    
    # --- Grouped View ---
    
    def _on_grouped_items_changed(self, grouped: Dict[str, List['CardItem']]):
        """Handle grouped items."""
        is_real_grouping = (
            self._viewmodel and 
            self._viewmodel.group_controller.is_active and 
            len(grouped) > 1
        )
        
        if is_real_grouping:
            self._is_grouped = True
            self._rebuild_grouped_view(grouped)
        else:
            # Single group = flat view
            all_items = []
            for items_list in grouped.values():
                all_items.extend(items_list)
            self.set_items(all_items)
    
    def _rebuild_grouped_view(self, grouped: Dict[str, List['CardItem']]):
        """Build grouped view with CardGroupWidgets."""
        # Clear everything first
        self._clear_visible_widgets()
        self._clear_groups()
        
        # Setup group layout
        if self._group_layout is None:
            self._group_layout = QVBoxLayout(self.container)
            self._group_layout.setContentsMargins(8, 8, 8, 8)
            self._group_layout.setSpacing(8)
            self._group_layout.addStretch()
        
        # Limit items per group for performance
        MAX_ITEMS = 50
        
        # Create groups
        for group_key, items in grouped.items():
            group = CardGroupWidget(group_key, self.container)
            group.set_title(f"{group_key} ({len(items)})")
            group.collapsed_changed.connect(self._on_group_collapsed)
            self._groups[group_key] = group
            
            # Create widgets directly (not from pool)
            for item in items[:MAX_ITEMS]:
                widget = self._create_group_widget(item, group.content_widget)
                group.add_item(widget)
            
            if len(items) > MAX_ITEMS:
                group.set_title(f"{group_key} ({len(items)}) - showing {MAX_ITEMS}")
            
            self._group_layout.insertWidget(self._group_layout.count() - 1, group)
        
        # Let layout handle sizing
        self.container.setMinimumHeight(0)
        self.container.setMaximumHeight(16777215)
        self.container.adjustSize()
        
        logger.info(f"CardView: Grouped into {len(grouped)} groups")
    
    def _clear_groups(self):
        """Clear all groups."""
        for group in self._groups.values():
            # Clear items (widgets will be deleted with group)
            group.clear_items()
            group.setParent(None)
            group.deleteLater()
        self._groups.clear()
    
    def _on_group_collapsed(self, group_key: str, collapsed: bool):
        """Handle group collapse/expand."""
        logger.debug(f"Group {group_key} collapsed: {collapsed}")
        # Content widget visibility is handled by CardGroupWidget
        # Just update scroll area
        QTimer.singleShot(10, lambda: self.container.adjustSize())
    
    # --- Selection ---
    
    def handle_item_click(self, widget: CardItemWidget, event):
        if not widget.data_context:
            return
        
        item_id = widget.data_context.id
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if item_id in self._selected_ids:
                self._deselect_item(item_id)
            else:
                self._select_item(item_id)
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            if self._last_selected_id:
                self._select_range(self._last_selected_id, item_id)
            else:
                self._select_item(item_id)
        else:
            self.clear_selection()
            self._select_item(item_id)
        
        self._last_selected_id = item_id
        self.selection_changed.emit(list(self._selected_ids))
    
    def _select_item(self, item_id: str):
        self._selected_ids.add(item_id)
        # Update visible widgets
        for widget in self._visible_widgets.values():
            if widget.data_context and widget.data_context.id == item_id:
                widget.set_selected(True)
        # Update group widgets
        for group in self._groups.values():
            for widget in group.item_widgets:
                if widget.data_context and widget.data_context.id == item_id:
                    widget.set_selected(True)
    
    def _deselect_item(self, item_id: str):
        self._selected_ids.discard(item_id)
        for widget in self._visible_widgets.values():
            if widget.data_context and widget.data_context.id == item_id:
                widget.set_selected(False)
        for group in self._groups.values():
            for widget in group.item_widgets:
                if widget.data_context and widget.data_context.id == item_id:
                    widget.set_selected(False)
    
    def _select_range(self, start_id: str, end_id: str):
        start_idx = end_idx = None
        for i, item in enumerate(self._items):
            if item.id == start_id:
                start_idx = i
            if item.id == end_id:
                end_idx = i
        if start_idx is None or end_idx is None:
            return
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        for i in range(start_idx, end_idx + 1):
            self._select_item(self._items[i].id)
    
    def clear_selection(self):
        for item_id in list(self._selected_ids):
            self._deselect_item(item_id)
        self._selected_ids.clear()
    
    def get_selected_items(self) -> List['CardItem']:
        return [item for item in self._items if item.id in self._selected_ids]
    
    # --- Card Size ---
    
    def set_card_size(self, width: int, height: int):
        self._card_width = width
        self._card_height = height
        
        if self._is_grouped:
            # Update grouped view widgets
            for group in self._groups.values():
                for widget in group.item_widgets:
                    widget.set_size(width, height)
                    widget.setFixedSize(width, height)
                # Force layout update
                group.items_layout.update_layout()
                group.content_widget.updateGeometry()
            self.container.adjustSize()
            logger.debug(f"CardView: Set card size {width}x{height} (grouped)")
        elif self._items:
            # Update flat view
            self.container.setFixedHeight(self._get_total_height())
            for idx, widget in self._visible_widgets.items():
                pos = self._get_item_position(idx)
                widget.move(pos)
                widget.set_size(width, height)
                widget.setFixedSize(width, height)
            self._visible_range = (-1, -1)
            self._update_timer.start(50)
            logger.debug(f"CardView: Set card size {width}x{height} (flat)")
    
    def set_thumbnail_size(self, size: Tuple[int, int]):
        for widget in self._visible_widgets.values():
            widget.set_thumbnail_size(size)
        for group in self._groups.values():
            for widget in group.item_widgets:
                widget.set_thumbnail_size(size)
    
    # --- Services ---
    
    def set_template_selector(self, selector):
        self._template_selector = selector
    
    def set_thumbnail_service(self, service):
        self._thumbnail_service = service
    
    def set_context_id(self, context_id: str):
        pass
    
    # --- Events ---
    
    def _on_scroll_changed(self, value: int):
        self._update_timer.start(16)
    
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if not self._is_grouped and self._items:
            self.container.setFixedHeight(self._get_total_height())
            for idx, widget in self._visible_widgets.items():
                pos = self._get_item_position(idx)
                widget.move(pos)
            self._visible_range = (-1, -1)
            self._update_timer.start(50)
