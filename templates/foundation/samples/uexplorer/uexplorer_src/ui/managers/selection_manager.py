"""
UExplorer - Selection Manager

Centralized selection tracking across all views.
Single source of truth for file/item selection state.
"""
from typing import List, Set, Optional
from dataclasses import dataclass, field
from bson import ObjectId
from PySide6.QtCore import QObject, Signal
from loguru import logger


@dataclass
class SelectionState:
    """Current selection state."""
    # Primary selection (files)
    selected_file_ids: Set[ObjectId] = field(default_factory=set)
    
    # Active/focused file (for preview)
    active_file_id: Optional[ObjectId] = None
    
    # Selection source (which view made the selection)
    source_view: str = ""  # "left_pane", "right_pane", "tree", etc.
    
    def count(self) -> int:
        """Get count of selected items."""
        return len(self.selected_file_ids)
    
    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return len(self.selected_file_ids) == 0
    
    def is_single(self) -> bool:
        """Check if exactly one item is selected."""
        return len(self.selected_file_ids) == 1
    
    def is_multiple(self) -> bool:
        """Check if multiple items are selected."""
        return len(self.selected_file_ids) > 1


class SelectionManager(QObject):
    """
    Centralized selection manager.
    
    Tracks file selections across all views and contexts.
    Propagates selection changes to all listening widgets.
    
    Usage:
        sel_mgr = SelectionManager()
        sel_mgr.selection_changed.connect(self.on_selection_changed)
        
        # Set selection
        sel_mgr.set_selection([file_id1, file_id2], source="left_pane")
        
        # Get selection
        selected = sel_mgr.get_selected_ids()
    """
    
    # Signals
    selection_changed = Signal()  # Any selection change
    active_changed = Signal(object)  # Active file ID changed (or None)
    count_changed = Signal(int)  # Selection count changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = SelectionState()
        logger.info("SelectionManager initialized")
    
    @property
    def state(self) -> SelectionState:
        """Get current selection state (read-only)."""
        return self._state
    
    def count(self) -> int:
        """Get count of selected items."""
        return self._state.count()
    
    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return self._state.is_empty()
    
    def get_selected_ids(self) -> List[ObjectId]:
        """Get list of selected file IDs."""
        return list(self._state.selected_file_ids)
    
    def get_active_id(self) -> Optional[ObjectId]:
        """Get active/focused file ID."""
        return self._state.active_file_id
    
    def get_source(self) -> str:
        """Get source view of last selection."""
        return self._state.source_view
    
    # ==================== Set Selection ====================
    
    def set_selection(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """
        Set selection (replaces current selection).
        
        Args:
            file_ids: List of file ObjectIds to select
            source: Identifier of the view making the selection
        """
        old_count = self._state.count()
        old_active = self._state.active_file_id
        
        self._state.selected_file_ids = set(file_ids)
        self._state.source_view = source
        
        # Update active to first selected item if changed
        if file_ids:
            self._state.active_file_id = file_ids[0]
        else:
            self._state.active_file_id = None
        
        # Emit signals
        self._emit_changed(old_count, old_active)
    
    def select_single(
        self,
        file_id: ObjectId,
        source: str = ""
    ) -> None:
        """Select single file (replaces selection)."""
        self.set_selection([file_id], source)
    
    def add_to_selection(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """Add files to current selection."""
        old_count = self._state.count()
        old_active = self._state.active_file_id
        
        for fid in file_ids:
            self._state.selected_file_ids.add(fid)
        self._state.source_view = source
        
        # Set active to last added
        if file_ids:
            self._state.active_file_id = file_ids[-1]
        
        self._emit_changed(old_count, old_active)
    
    def remove_from_selection(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """Remove files from current selection."""
        old_count = self._state.count()
        old_active = self._state.active_file_id
        
        for fid in file_ids:
            self._state.selected_file_ids.discard(fid)
        self._state.source_view = source
        
        # Clear active if removed
        if old_active in file_ids:
            if self._state.selected_file_ids:
                self._state.active_file_id = next(iter(self._state.selected_file_ids))
            else:
                self._state.active_file_id = None
        
        self._emit_changed(old_count, old_active)
    
    def toggle_selection(
        self,
        file_id: ObjectId,
        source: str = ""
    ) -> bool:
        """
        Toggle file selection state.
        
        Returns:
            True if file is now selected, False if deselected
        """
        if file_id in self._state.selected_file_ids:
            self.remove_from_selection([file_id], source)
            return False
        else:
            self.add_to_selection([file_id], source)
            return True
    
    def select_all(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """Select all given files."""
        self.set_selection(file_ids, source)
    
    def clear_selection(self, source: str = "") -> None:
        """Clear all selections."""
        self.set_selection([], source)
    
    # ==================== Active File ====================
    
    def set_active(
        self,
        file_id: Optional[ObjectId],
        source: str = ""
    ) -> None:
        """Set active/focused file without changing selection."""
        old_active = self._state.active_file_id
        self._state.active_file_id = file_id
        self._state.source_view = source
        
        if old_active != file_id:
            self.active_changed.emit(file_id)
    
    # ==================== Range Selection ====================
    
    def select_range(
        self,
        all_ids: List[ObjectId],
        anchor_id: ObjectId,
        target_id: ObjectId,
        source: str = ""
    ) -> None:
        """
        Select range of files between anchor and target.
        
        Args:
            all_ids: Ordered list of all file IDs in view
            anchor_id: Starting point of range
            target_id: End point of range
            source: Source view
        """
        try:
            anchor_idx = all_ids.index(anchor_id)
            target_idx = all_ids.index(target_id)
            
            start = min(anchor_idx, target_idx)
            end = max(anchor_idx, target_idx) + 1
            
            range_ids = all_ids[start:end]
            self.set_selection(range_ids, source)
            
        except ValueError:
            # anchor or target not found
            self.select_single(target_id, source)
    
    # ==================== Internal ====================
    
    def _emit_changed(
        self,
        old_count: int,
        old_active: Optional[ObjectId]
    ) -> None:
        """Emit appropriate signals."""
        new_count = self._state.count()
        
        self.selection_changed.emit()
        
        if new_count != old_count:
            self.count_changed.emit(new_count)
        
        if self._state.active_file_id != old_active:
            self.active_changed.emit(self._state.active_file_id)
        
        logger.debug(f"Selection: count={new_count}, source={self._state.source_view}")
