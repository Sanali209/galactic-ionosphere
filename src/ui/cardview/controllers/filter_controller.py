"""
FilterController - Manages filtering logic for CardView.

Supports text search, predicates, and combinable filters.
"""
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class FilterController:
    """
    Controls filtering for CardView.
    
    Features:
    - Text search across fields
    - Custom predicate filters
    - Named filters (combinable with AND/OR)
    - Quick filters (rating, type, etc.)
    
    Example:
        controller = FilterController()
        controller.set_text_filter("vacation")
        controller.add_predicate("rating", lambda item: item.rating >= 4)
        filtered = controller.apply(items)
    """
    
    def __init__(self):
        """Initialize filter controller."""
        self._text_filter: Optional[str] = None
        self._text_fields: Optional[List[str]] = None
        self._predicates: Dict[str, Callable[['CardItem'], bool]] = {}
        self._combine_mode: str = "and"  # "and" or "or"
    
    # --- Text Filter ---
    
    def set_text_filter(self, text: str, fields: Optional[List[str]] = None):
        """
        Set text search filter.
        
        Args:
            text: Search text (case-insensitive)
            fields: Fields to search (default: title, subtitle, tags)
        """
        self._text_filter = text.strip() if text else None
        self._text_fields = fields
        logger.debug(f"Text filter set: '{self._text_filter}'")
    
    def clear_text_filter(self):
        """Clear text filter."""
        self._text_filter = None
        self._text_fields = None
    
    @property
    def text_filter(self) -> Optional[str]:
        """Current text filter."""
        return self._text_filter
    
    # --- Predicate Filters ---
    
    def add_predicate(self, name: str, predicate: Callable[['CardItem'], bool]):
        """
        Add named predicate filter.
        
        Args:
            name: Unique name for this filter
            predicate: Function(item) returning True to include
        """
        self._predicates[name] = predicate
        logger.debug(f"Predicate '{name}' added")
    
    def remove_predicate(self, name: str):
        """
        Remove named predicate.
        
        Args:
            name: Name of predicate to remove
        """
        if name in self._predicates:
            del self._predicates[name]
            logger.debug(f"Predicate '{name}' removed")
    
    def clear_predicates(self):
        """Remove all predicates."""
        self._predicates.clear()
    
    # --- Quick Filters ---
    
    def filter_by_rating(self, min_rating: int):
        """Filter by minimum rating."""
        self.add_predicate("rating", lambda item: item.rating >= min_rating)
    
    def filter_by_type(self, item_type: str):
        """Filter by item type."""
        self.add_predicate("type", lambda item: item.item_type == item_type)
    
    def filter_by_tags(self, tags: List[str], match_all: bool = False):
        """
        Filter by tags.
        
        Args:
            tags: Tags to match
            match_all: True to require all tags, False for any
        """
        def predicate(item):
            if not item.tags:
                return False
            if match_all:
                return all(t in item.tags for t in tags)
            return any(t in item.tags for t in tags)
        
        self.add_predicate("tags", predicate)
    
    # --- Combine Mode ---
    
    def set_combine_mode(self, mode: str):
        """
        Set how predicates are combined.
        
        Args:
            mode: "and" (all must match) or "or" (any must match)
        """
        if mode not in ("and", "or"):
            raise ValueError("Mode must be 'and' or 'or'")
        self._combine_mode = mode
    
    # --- Apply ---
    
    def apply(self, items: List['CardItem']) -> List['CardItem']:
        """
        Apply all filters to items.
        
        Args:
            items: Items to filter
            
        Returns:
            Filtered list
        """
        if not items:
            return []
        
        result = items
        
        # Apply text filter first
        if self._text_filter:
            result = [
                item for item in result
                if item.matches_text(self._text_filter, self._text_fields)
            ]
        
        # Apply predicates
        if self._predicates:
            result = [
                item for item in result
                if self._check_predicates(item)
            ]
        
        return result
    
    def _check_predicates(self, item: 'CardItem') -> bool:
        """Check if item passes all/any predicates."""
        if not self._predicates:
            return True
        
        results = [pred(item) for pred in self._predicates.values()]
        
        if self._combine_mode == "and":
            return all(results)
        return any(results)
    
    # --- State ---
    
    @property
    def is_active(self) -> bool:
        """Check if any filter is active."""
        return bool(self._text_filter or self._predicates)
    
    @property
    def active_filter_names(self) -> List[str]:
        """Get names of active predicate filters."""
        names = list(self._predicates.keys())
        if self._text_filter:
            names.insert(0, "text")
        return names
    
    def clear(self):
        """Clear all filters."""
        self._text_filter = None
        self._text_fields = None
        self._predicates.clear()
