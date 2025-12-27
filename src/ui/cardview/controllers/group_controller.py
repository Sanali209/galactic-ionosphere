"""
GroupController - Manages grouping logic for CardView.

Groups items by field value with collapse state tracking.
"""
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.ui.cardview.models.card_item import CardItem


class GroupController:
    """
    Controls grouping for CardView.
    
    Features:
    - Group by field value
    - Custom group key function
    - Collapse state tracking
    - Group ordering
    
    Example:
        controller = GroupController()
        controller.set_group_field("item_type")
        grouped = controller.apply(items)  # {"image": [...], "document": [...]}
    """
    
    def __init__(self):
        """Initialize group controller."""
        self._group_field: Optional[str] = None
        self._group_key_fn: Optional[Callable[['CardItem'], str]] = None
        self._collapsed_groups: Set[str] = set()
        self._group_order: Optional[List[str]] = None
    
    # --- Field Grouping ---
    
    def set_group_field(self, field: str):
        """
        Set grouping by field.
        
        Args:
            field: Field name to group by
        """
        self._group_field = field
        self._group_key_fn = None
        logger.debug(f"Grouping by field: {field}")
    
    @property
    def group_field(self) -> Optional[str]:
        """Current group field."""
        return self._group_field
    
    # --- Custom Key Function ---
    
    def set_group_key_function(self, fn: Callable[['CardItem'], str]):
        """
        Set custom group key function.
        
        Args:
            fn: Function(item) returning group key string
        """
        self._group_key_fn = fn
        self._group_field = None
        logger.debug("Custom group key function set")
    
    # --- Apply ---
    
    def apply(self, items: List['CardItem']) -> Dict[str, List['CardItem']]:
        """
        Group items by current settings.
        
        Args:
            items: Items to group
            
        Returns:
            Dict mapping group key to list of items
        """
        if not items:
            return {}
        
        if not self._group_field and not self._group_key_fn:
            # No grouping, return single group
            return {"all": items}
        
        grouped: Dict[str, List['CardItem']] = {}
        
        for item in items:
            key = self._get_group_key(item)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        
        # Apply custom order if set
        if self._group_order:
            result = {}
            for key in self._group_order:
                if key in grouped:
                    result[key] = grouped.pop(key)
            # Add remaining groups
            result.update(grouped)
            return result
        
        # Default: sort groups alphabetically
        return dict(sorted(grouped.items()))
    
    def _get_group_key(self, item: 'CardItem') -> str:
        """Get group key for item."""
        if self._group_key_fn:
            return self._group_key_fn(item)
        
        if self._group_field:
            value = item.get_field(self._group_field)
            return str(value) if value is not None else "Other"
        
        return "all"
    
    # --- Collapse State ---
    
    def is_collapsed(self, group_key: str) -> bool:
        """Check if group is collapsed."""
        return group_key in self._collapsed_groups
    
    def set_collapsed(self, group_key: str, collapsed: bool):
        """
        Set collapse state for group.
        
        Args:
            group_key: Group to update
            collapsed: True to collapse
        """
        if collapsed:
            self._collapsed_groups.add(group_key)
        else:
            self._collapsed_groups.discard(group_key)
    
    def toggle_collapsed(self, group_key: str):
        """Toggle collapse state."""
        if group_key in self._collapsed_groups:
            self._collapsed_groups.remove(group_key)
        else:
            self._collapsed_groups.add(group_key)
    
    def collapse_all(self):
        """Collapse all groups."""
        # Will be populated as groups are applied
        pass
    
    def expand_all(self):
        """Expand all groups."""
        self._collapsed_groups.clear()
    
    @property
    def collapsed_groups(self) -> List[str]:
        """Get list of collapsed group keys."""
        return list(self._collapsed_groups)
    
    def set_collapsed_groups(self, groups: List[str]):
        """Restore collapsed groups from saved state."""
        self._collapsed_groups = set(groups)
    
    # --- Group Order ---
    
    def set_group_order(self, order: List[str]):
        """
        Set custom group display order.
        
        Args:
            order: List of group keys in display order
        """
        self._group_order = order
    
    # --- Clear ---
    
    def clear(self):
        """Clear grouping settings."""
        self._group_field = None
        self._group_key_fn = None
        self._collapsed_groups.clear()
        self._group_order = None
    
    @property
    def is_active(self) -> bool:
        """Check if grouping is active."""
        return bool(self._group_field or self._group_key_fn)
