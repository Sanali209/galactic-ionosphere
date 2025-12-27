"""
CardViewModel - MVVM ViewModel for CardView.

Controls behavior for CardView through Sort/Filter/Group controllers.
"""
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from PySide6.QtCore import QObject, Signal
from loguru import logger

from src.ui.mvvm import BaseViewModel, BindableProperty
from src.ui.cardview.controllers.sort_controller import SortController
from src.ui.cardview.controllers.filter_controller import FilterController
from src.ui.cardview.controllers.group_controller import GroupController

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class CardViewModel(BaseViewModel):
    """
    ViewModel for CardView with full MVVM integration.
    
    Controls:
    - Data source binding
    - Sort/Filter/Grouping behavior
    - Selection state
    - State persistence
    
    Properties (Bindable):
        items: Source items
        visible_items: After sort/filter
        grouped_items: Dict[group_key, List[items]]
        selected_ids: Currently selected
        thumbnail_size: Current size
    
    Example:
        vm = CardViewModel(locator)
        await vm.load_items(items)
        vm.sort_by_field("title")
        vm.filter_by_text("vacation")
        vm.group_by_field("item_type")
    """
    
    # Signals
    itemsChanged = Signal(list)
    visibleItemsChanged = Signal(list)
    groupedItemsChanged = Signal(dict)
    selectionChanged = Signal(list)
    thumbnailSizeChanged = Signal(int)
    
    # Bindable properties
    items = BindableProperty(default=[])
    visible_items = BindableProperty(default=[])
    grouped_items = BindableProperty(default={})
    selected_ids = BindableProperty(default=[])
    thumbnail_size = BindableProperty(default=200)
    
    def __init__(self, locator=None):
        """
        Initialize CardViewModel.
        
        Args:
            locator: ServiceLocator for accessing services
        """
        super().__init__(locator)
        
        # Controllers
        self.sort_controller = SortController()
        self.filter_controller = FilterController()
        self.group_controller = GroupController()
        
        # State controller (initialized lazily)
        self._state_controller = None
        
        # Data source
        self._data_source = None
        
        # Internal items list
        self._items: List['CardItem'] = []
    
    # --- State Controller (Lazy Init) ---
    
    @property
    def state_controller(self):
        """Get or create state controller."""
        if self._state_controller is None:
            from src.ui.cardview.controllers.state_controller import StateController
            self._state_controller = StateController(self, self._locator)
        return self._state_controller
    
    # --- Data Loading ---
    
    async def load_items(self, items: List['CardItem']):
        """
        Load items from data source.
        
        Args:
            items: List of CardItem to display
        """
        self._items = items
        self.items = items
        self._apply_transformations()
        logger.debug(f"Loaded {len(items)} items")
    
    def set_items(self, items: List['CardItem']):
        """
        Set items synchronously.
        
        Args:
            items: List of CardItem to display
        """
        self._items = items
        self.items = items
        self._apply_transformations()
    
    def get_item_by_id(self, item_id: str) -> Optional['CardItem']:
        """
        Find item by ID.
        
        Args:
            item_id: Item ID to find
            
        Returns:
            CardItem or None if not found
        """
        for item in self._items:
            if item.id == item_id:
                return item
        return None
    
    async def refresh(self):
        """Refresh from data source."""
        if self._data_source:
            items = await self._data_source.fetch()
            await self.load_items(items)
    
    # --- Sort ---
    
    def sort_by_field(self, field: str, ascending: bool = True):
        """
        Sort by item field.
        
        Args:
            field: Field name to sort by
            ascending: Sort direction
        """
        self.sort_controller.sort_by_field(field, ascending)
        self._apply_transformations()
    
    def sort_by_custom(self, comparator: Callable[['CardItem', 'CardItem'], int]):
        """
        Sort with custom comparator.
        
        Args:
            comparator: Function(a, b) returning -1, 0, or 1
        """
        self.sort_controller.set_custom_comparator(comparator)
        self._apply_transformations()
    
    def set_user_sort(self, order: List[str]):
        """
        Apply user-defined order (persisted).
        
        Args:
            order: List of item IDs in desired order
        """
        self.sort_controller.set_user_order(order)
        self._apply_transformations()
    
    def apply_algorithm_sort(
        self,
        algorithm: Callable[[List['CardItem']], List['CardItem']]
    ) -> str:
        """
        Apply heavy sorting algorithm and get hash.
        
        Args:
            algorithm: Sorting function
            
        Returns:
            Hash of resulting order (for persistence)
        """
        sorted_items, hash_value = self.sort_controller.apply_algorithm_sort(
            algorithm, self._items
        )
        self._apply_transformations()
        return hash_value
    
    # --- Filter ---
    
    def filter_by_text(self, text: str, fields: Optional[List[str]] = None):
        """
        Filter by search text.
        
        Args:
            text: Search text
            fields: Fields to search (default: title, subtitle, tags)
        """
        self.filter_controller.set_text_filter(text, fields)
        self._apply_transformations()
    
    def filter_by_predicate(
        self,
        name: str,
        predicate: Callable[['CardItem'], bool]
    ):
        """
        Add predicate filter.
        
        Args:
            name: Unique name for filter
            predicate: Function(item) returning True to include
        """
        self.filter_controller.add_predicate(name, predicate)
        self._apply_transformations()
    
    def clear_filter(self):
        """Clear all filters."""
        self.filter_controller.clear()
        self._apply_transformations()
    
    # --- Grouping ---
    
    def group_by_field(self, field: str):
        """
        Group items by field value.
        
        Args:
            field: Field name to group by
        """
        self.group_controller.set_group_field(field)
        self._apply_transformations()
    
    def set_group_key_function(self, fn: Callable[['CardItem'], str]):
        """
        Set custom grouping function.
        
        Args:
            fn: Function(item) returning group key
        """
        self.group_controller.set_group_key_function(fn)
        self._apply_transformations()
    
    def clear_grouping(self):
        """Remove grouping."""
        self.group_controller.clear()
        self._apply_transformations()
    
    # --- Selection ---
    
    def select_items(self, ids: List[str]):
        """Select items by ID."""
        self.selected_ids = ids
        self.selectionChanged.emit(ids)
    
    def clear_selection(self):
        """Clear selection."""
        self.selected_ids = []
        self.selectionChanged.emit([])
    
    def get_selected_items(self) -> List['CardItem']:
        """Get selected items."""
        selected_set = set(self.selected_ids)
        return [item for item in self._items if item.id in selected_set]
    
    # --- Thumbnail Size ---
    
    def set_thumbnail_size(self, size: int):
        """
        Set thumbnail size.
        
        Args:
            size: Thumbnail width (height same)
        """
        self.thumbnail_size = size
        self.thumbnailSizeChanged.emit(size)
    
    # --- Transformations ---
    
    def _apply_transformations(self):
        """Apply sort → filter → group pipeline."""
        result = self._items
        
        # Sort
        result = self.sort_controller.apply(result)
        
        # Filter
        result = self.filter_controller.apply(result)
        
        # Group
        grouped = self.group_controller.apply(result)
        self.grouped_items = grouped
        
        # Emit signals - grouped FIRST when grouping is active
        # This ensures CardView knows to use grouped mode before receiving items
        if self.group_controller.is_active:
            self.groupedItemsChanged.emit(grouped)
        else:
            # No grouping - emit visible items for flat view
            self.visible_items = result
            self.visibleItemsChanged.emit(result)
        
        logger.debug(
            f"Transformations applied: {len(self._items)} → {len(result)} items, "
            f"{len(grouped)} groups"
        )
    
    # --- State Persistence ---
    
    async def save_state(self, context_id: str):
        """Save current state for context."""
        await self.state_controller.save_state(context_id, {
            "user_sort_order": self.sort_controller.get_user_order(),
            "algorithm_hash": self.sort_controller.current_hash,
            "collapsed_groups": self.group_controller.collapsed_groups,
            "thumbnail_size": self.thumbnail_size,
        })
    
    async def restore_state(self, context_id: str):
        """Restore state for context."""
        state = await self.state_controller.load_state(context_id)
        if state:
            if state.get("user_sort_order"):
                self.set_user_sort(state["user_sort_order"])
            if state.get("collapsed_groups"):
                self.group_controller.set_collapsed_groups(state["collapsed_groups"])
            if state.get("thumbnail_size"):
                self.set_thumbnail_size(state["thumbnail_size"])
