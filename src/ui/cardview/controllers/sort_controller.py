"""
SortController - Manages sorting logic for CardView.

Supports field-based sorting, custom comparators, and user sort persistence.
"""
import hashlib
from typing import Any, Callable, List, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem, SortOrder


class SortController:
    """
    Controls sorting for CardView.
    
    Features:
    - Sort by field (ascending/descending)
    - Custom comparator functions
    - User-defined order (drag reorder)
    - Algorithm result hashing for persistence
    
    Example:
        controller = SortController()
        controller.sort_by_field("title", ascending=True)
        sorted_items = controller.apply(items)
    """
    
    def __init__(self):
        """Initialize sort controller."""
        self._sort_field: Optional[str] = None
        self._ascending: bool = True
        self._custom_comparator: Optional[Callable] = None
        self._user_order: Optional[List[str]] = None
        self._current_hash: Optional[str] = None
    
    # --- Field Sorting ---
    
    def sort_by_field(self, field: str, ascending: bool = True):
        """
        Set sorting by field.
        
        Args:
            field: Field name to sort by
            ascending: True for ascending, False for descending
        """
        self._sort_field = field
        self._ascending = ascending
        self._custom_comparator = None
        self._user_order = None
        logger.debug(f"Sort set: {field} {'asc' if ascending else 'desc'}")
    
    @property
    def sort_field(self) -> Optional[str]:
        """Current sort field."""
        return self._sort_field
    
    @property
    def ascending(self) -> bool:
        """Current sort direction."""
        return self._ascending
    
    # --- Custom Comparator ---
    
    def set_custom_comparator(self, comparator: Callable[['CardItem', 'CardItem'], int]):
        """
        Set custom comparator function.
        
        Args:
            comparator: Function(a, b) returning -1, 0, or 1
        """
        self._custom_comparator = comparator
        self._sort_field = None
        self._user_order = None
        logger.debug("Custom comparator set")
    
    # --- User Sort ---
    
    def set_user_order(self, order: List[str]):
        """
        Set user-defined order by item IDs.
        
        Args:
            order: List of item IDs in desired order
        """
        self._user_order = order
        self._sort_field = None
        self._custom_comparator = None
        logger.debug(f"User order set: {len(order)} items")
    
    def get_user_order(self) -> Optional[List[str]]:
        """Get current user order."""
        return self._user_order
    
    # --- Apply ---
    
    def apply(self, items: List['CardItem']) -> List['CardItem']:
        """
        Apply current sort to items.
        
        Args:
            items: Items to sort
            
        Returns:
            Sorted list (new list, original unchanged)
        """
        if not items:
            return []
        
        result = items.copy()
        
        if self._user_order:
            result = self._apply_user_order(result)
        elif self._custom_comparator:
            result = self._apply_custom(result)
        elif self._sort_field:
            result = self._apply_field_sort(result)
        
        return result
    
    def _apply_field_sort(self, items: List['CardItem']) -> List['CardItem']:
        """Sort by field."""
        def get_key(item):
            value = item.get_field(self._sort_field)
            # Handle None values
            if value is None:
                return ("", "") if isinstance(value, str) else (1, 0)
            # Case-insensitive string sort
            if isinstance(value, str):
                return (0, value.lower())
            return (0, value)
        
        return sorted(items, key=get_key, reverse=not self._ascending)
    
    def _apply_custom(self, items: List['CardItem']) -> List['CardItem']:
        """Sort with custom comparator."""
        from functools import cmp_to_key
        return sorted(items, key=cmp_to_key(self._custom_comparator))
    
    def _apply_user_order(self, items: List['CardItem']) -> List['CardItem']:
        """Sort by user-defined order."""
        order_map = {id: idx for idx, id in enumerate(self._user_order)}
        return sorted(
            items,
            key=lambda x: order_map.get(x.id, float('inf'))
        )
    
    # --- Algorithm Hashing ---
    
    def apply_algorithm_sort(
        self,
        algorithm: Callable[[List['CardItem']], List['CardItem']],
        items: List['CardItem']
    ) -> tuple[List['CardItem'], str]:
        """
        Apply heavy sorting algorithm and cache result hash.
        
        Args:
            algorithm: Sorting function
            items: Items to sort
            
        Returns:
            Tuple of (sorted items, hash of order)
        """
        sorted_items = algorithm(items)
        
        # Create hash of sort order
        order_str = ":".join(item.id for item in sorted_items)
        order_hash = hashlib.md5(order_str.encode()).hexdigest()[:16]
        
        self._current_hash = order_hash
        self._user_order = [item.id for item in sorted_items]
        
        logger.debug(f"Algorithm sort applied, hash: {order_hash}")
        return sorted_items, order_hash
    
    @property
    def current_hash(self) -> Optional[str]:
        """Get hash of current sort order."""
        return self._current_hash
    
    def restore_from_order(
        self,
        items: List['CardItem'],
        saved_order: List[str]
    ) -> List['CardItem']:
        """
        Restore order from saved IDs (skip re-running algorithm).
        
        Args:
            items: Items to reorder
            saved_order: Previously saved ID order
            
        Returns:
            Reordered items
        """
        self._user_order = saved_order
        return self._apply_user_order(items)
    
    # --- Reset ---
    
    def clear(self):
        """Clear all sorting settings."""
        self._sort_field = None
        self._ascending = True
        self._custom_comparator = None
        self._user_order = None
        self._current_hash = None
