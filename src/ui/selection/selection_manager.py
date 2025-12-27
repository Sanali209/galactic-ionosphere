"""
Selection Manager - Cross-view selection tracking.

Provides centralized selection state that can be shared across
multiple views. Single source of truth for what items are selected.
"""
from typing import List, Set, Optional, Any
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal
from loguru import logger

from src.core.base_system import BaseSystem


@dataclass
class SelectionState:
    """
    Immutable snapshot of current selection.
    
    Attributes:
        selected_ids: Set of selected item IDs
        active_id: Currently focused/active item
        source: View that made the selection
    """
    selected_ids: Set[Any] = field(default_factory=set)
    active_id: Optional[Any] = None
    source: str = ""
    
    @property
    def count(self) -> int:
        """Get count of selected items."""
        return len(self.selected_ids)
    
    @property
    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return len(self.selected_ids) == 0
    
    @property
    def is_single(self) -> bool:
        """Check if exactly one item is selected."""
        return len(self.selected_ids) == 1
    
    @property
    def is_multiple(self) -> bool:
        """Check if multiple items are selected."""
        return len(self.selected_ids) > 1
    
    def contains(self, item_id: Any) -> bool:
        """Check if item is selected."""
        return item_id in self.selected_ids


class _SelectionSignals(QObject):
    """Signal holder to avoid metaclass conflict."""
    selection_changed = Signal(object)  # SelectionState
    active_changed = Signal(object)  # active_id or None
    count_changed = Signal(int)  # selection count


class SelectionManager(BaseSystem):
    """
    Centralized selection manager.
    
    Tracks item selections across all views. Views connect to signals
    to synchronize their visual selection state.
    
    Usage:
        sel_mgr = locator.get_system(SelectionManager)
        
        # Set selection
        sel_mgr.set_selection([id1, id2], source="tree_view")
        
        # Listen for changes via signals property
        sel_mgr.signals.selection_changed.connect(on_selection_changed)
        
        # Check state
        state = sel_mgr.state
        if state.is_single:
            show_details(state.active_id)
    """
    
    def __init__(self, locator, config):
        """Initialize SelectionManager."""
        super().__init__(locator, config)
        
        # Qt signals via composition
        self._signals = _SelectionSignals()
        
        self._selected_ids: Set[Any] = set()
        self._active_id: Optional[Any] = None
        self._source: str = ""
    
    @property
    def signals(self) -> _SelectionSignals:
        """Get Qt signals object for UI binding."""
        return self._signals
    
    async def initialize(self):
        """Initialize the SelectionManager."""
        await super().initialize()
        logger.info("SelectionManager initialized")
    
    async def shutdown(self):
        """Shutdown and clear selection."""
        self.clear()
        await super().shutdown()
    
    @property
    def state(self) -> SelectionState:
        """Get current selection state (read-only snapshot)."""
        return SelectionState(
            selected_ids=self._selected_ids.copy(),
            active_id=self._active_id,
            source=self._source
        )
    
    @property
    def count(self) -> int:
        """Get count of selected items."""
        return len(self._selected_ids)
    
    @property
    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return len(self._selected_ids) == 0
    
    def get_selected_ids(self) -> List[Any]:
        """Get list of selected IDs."""
        return list(self._selected_ids)
    
    def get_active_id(self) -> Optional[Any]:
        """Get active/focused item ID."""
        return self._active_id
    
    def set_selection(self, ids: List[Any], source: str = "") -> None:
        """
        Set selection (replaces current).
        
        Args:
            ids: List of item IDs to select
            source: Identifier of the view making the selection
        """
        old_count = len(self._selected_ids)
        old_active = self._active_id
        
        self._selected_ids = set(ids)
        self._source = source
        
        # Set active to first selected if not already in selection
        if ids and (self._active_id not in self._selected_ids):
            self._active_id = ids[0]
        elif not ids:
            self._active_id = None
        
        self._emit_changes(old_count, old_active)
        logger.debug(f"Selection set: {len(ids)} items from {source}")
    
    def select_single(self, item_id: Any, source: str = "") -> None:
        """Select single item (replaces selection)."""
        self.set_selection([item_id], source)
    
    def add_to_selection(self, ids: List[Any], source: str = "") -> None:
        """Add items to current selection."""
        old_count = len(self._selected_ids)
        old_active = self._active_id
        
        self._selected_ids.update(ids)
        self._source = source
        
        if ids and self._active_id is None:
            self._active_id = ids[0]
        
        self._emit_changes(old_count, old_active)
    
    def remove_from_selection(self, ids: List[Any], source: str = "") -> None:
        """Remove items from selection."""
        old_count = len(self._selected_ids)
        old_active = self._active_id
        
        self._selected_ids -= set(ids)
        self._source = source
        
        # Clear active if removed
        if self._active_id in ids:
            remaining = list(self._selected_ids)
            self._active_id = remaining[0] if remaining else None
        
        self._emit_changes(old_count, old_active)
    
    def toggle(self, item_id: Any, source: str = "") -> bool:
        """
        Toggle item selection.
        
        Returns:
            True if now selected, False if deselected
        """
        if item_id in self._selected_ids:
            self.remove_from_selection([item_id], source)
            return False
        else:
            self.add_to_selection([item_id], source)
            return True
    
    def clear(self, source: str = "") -> None:
        """Clear all selections."""
        if self._selected_ids:
            self.set_selection([], source)
    
    def set_active(self, item_id: Optional[Any], source: str = "") -> None:
        """Set active item without changing selection."""
        if self._active_id != item_id:
            self._active_id = item_id
            self._source = source
            self._signals.active_changed.emit(item_id)
    
    def select_range(self, all_ids: List[Any], start_id: Any, 
                     end_id: Any, source: str = "") -> None:
        """
        Select range of items between start and end.
        
        Args:
            all_ids: Ordered list of all item IDs
            start_id: Range start
            end_id: Range end
            source: Source view
        """
        try:
            start_idx = all_ids.index(start_id)
            end_idx = all_ids.index(end_id)
            
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            range_ids = all_ids[start_idx:end_idx + 1]
            self.set_selection(range_ids, source)
            
        except ValueError:
            logger.warning("Range selection failed: ID not found")
    
    def _emit_changes(self, old_count: int, old_active: Optional[Any]) -> None:
        """Emit appropriate signals based on what changed."""
        new_state = self.state
        self._signals.selection_changed.emit(new_state)
        
        if len(self._selected_ids) != old_count:
            self._signals.count_changed.emit(len(self._selected_ids))
        
        if self._active_id != old_active:
            self._signals.active_changed.emit(self._active_id)
