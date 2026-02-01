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


from src.ui.mvvm.viewmodel import BaseViewModel
from src.ui.mvvm.bindable import BindableProperty, BindableBase, BindableList


@dataclass
class SelectionState:
    """Snapshot of selection state for serialization or legacy sync."""
    selected_file_ids: Set[ObjectId] = field(default_factory=set)
    active_file_id: Optional[ObjectId] = None
    source_view: str = ""

    def count(self) -> int:
        return len(self.selected_file_ids)


class SelectionManager(BaseViewModel):
    """
    Centralized selection manager using Reactive MVVM patterns.
    
    Tracks file selections across all views and contexts.
    Uses 'sync channels' to stay in sync with other components.
    """
    
    # Reactive properties (Single Source of Truth)
    active_file_id = BindableProperty(sync_channel="active_file")
    selected_file_ids = BindableProperty(sync_channel="selection")
    
    # Internal context (not synced)
    selection_count = BindableProperty(default=0)
    
    # Specific changed signals for reactive properties (required by PySide6 for connection)
    active_file_idChanged = Signal(object)
    selected_file_idsChanged = Signal(object)
    
    # Legacy Signals (for backward compatibility with old widgets)
    selection_changed = Signal()
    active_changed = Signal(object)
    count_changed = Signal(int)
    
    def __init__(self, parent=None):
        """
        Initialize the Selection Manager.
        
        Args:
            parent: Usually MainWindow, used to extract ServiceLocator.
        """
        # Extract locator if parent is MainWindow
        locator = getattr(parent, 'locator', None)
        super().__init__(locator)
        
        # Initialize tracked state
        self.selected_file_ids = BindableList()
        # self.active_file_id is handled by BindableProperty
        self.source_view = ""
        
        # Internal reactive connections:
        # When active_file_id changes (even locally), notify legacy listeners
        self.active_file_idChanged.connect(lambda val: self.active_changed.emit(val))
        
        # When collection changes, update count and legacy signals
        self.selected_file_ids.collectionChanged.connect(self._on_collection_changed)
        
        # Register for synchronization via ContextSyncManager
        self.initialize_reactivity()
        
        logger.info("SelectionManager (Reactive) initialized")

    def _on_collection_changed(self, items):
        """Update count and emit legacy signals when selection list changes."""
        self.selection_count = len(items)
        self.count_changed.emit(self.selection_count)
        self.selection_changed.emit()

    @property
    def state(self) -> SelectionState:
        """Legacy compatibility: returns a snapshot of the state."""
        return SelectionState(
            selected_file_ids=set(self.selected_file_ids),
            active_file_id=self.active_file_id,
            source_view=self.source_view
        )

    def count(self) -> int:
        return self.selection_count
    
    def is_empty(self) -> bool:
        return self.selection_count == 0
    
    def get_selected_ids(self) -> List[ObjectId]:
        return list(self.selected_file_ids)
    
    def get_active_id(self) -> Optional[ObjectId]:
        return self.active_file_id
    
    def get_source(self) -> str:
        return self.source_view

    # ==================== Set Selection ====================
    
    def set_selection(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """Set selection (replaces current selection)."""
        self.source_view = source
        
        # Update reactive list
        self.selected_file_ids.clear()
        if file_ids:
            self.selected_file_ids.extend(file_ids)
            self.active_file_id = file_ids[0]
        else:
            self.active_file_id = None
            
        # Notify legacy listeners
        self.active_changed.emit(self.active_file_id)
    
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
        self.source_view = source
        
        added = False
        for fid in file_ids:
            if fid not in self.selected_file_ids:
                self.selected_file_ids.append(fid)
                added = True
        
        # Set active to last added
        if file_ids:
            self.active_file_id = file_ids[-1]
    
    def remove_from_selection(
        self,
        file_ids: List[ObjectId],
        source: str = ""
    ) -> None:
        """Remove files from current selection."""
        self.source_view = source
        
        removed = False
        for fid in file_ids:
            if fid in self.selected_file_ids:
                self.selected_file_ids.remove(fid)
                removed = True
        
        # Clear active if removed
        if self.active_file_id in file_ids:
            if self.selected_file_ids:
                self.active_file_id = self.selected_file_ids[0]
            else:
                self.active_file_id = None
    
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
        if file_id in self.selected_file_ids:
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
        self.active_file_id = file_id
        self.source_view = source
    
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
    
    def _on_mode_changed(self, mode):
        """Reserved for future use."""
        pass
