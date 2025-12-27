"""
UExplorer - Filter Manager

Centralized filtering logic for file views.
Single source of truth for all filter state.
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from bson import ObjectId
from PySide6.QtCore import QObject, Signal
from loguru import logger


@dataclass
class FilterState:
    """Current filter state."""
    # Tag filtering
    selected_tag_ids: Set[ObjectId] = field(default_factory=set)
    tag_mode: str = "any"  # any, all, none
    
    # Album filtering
    selected_album_id: Optional[ObjectId] = None
    
    # File type filtering
    file_types: Set[str] = field(default_factory=set)  # empty = all
    
    # Rating filtering
    min_rating: int = 0
    max_rating: int = 5
    
    # Label filtering
    labels: Set[str] = field(default_factory=set)
    
    # Text search
    search_text: str = ""
    
    # Path filtering
    directory_path: Optional[str] = None
    include_subdirs: bool = True
    
    def is_active(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.selected_tag_ids or
            self.selected_album_id or
            self.file_types or
            self.labels or
            self.search_text or
            self.min_rating > 0 or
            self.directory_path
        )
    
    def to_mongo_query(self) -> Dict[str, Any]:
        """Convert filter state to MongoDB query."""
        query = {}
        
        # Tag filtering
        if self.selected_tag_ids:
            if self.tag_mode == "all":
                query["tag_ids"] = {"$all": list(self.selected_tag_ids)}
            elif self.tag_mode == "none":
                query["tag_ids"] = {"$nin": list(self.selected_tag_ids)}
            else:  # any
                query["tag_ids"] = {"$in": list(self.selected_tag_ids)}
        
        # File type
        if self.file_types:
            query["file_type"] = {"$in": list(self.file_types)}
        
        # Rating
        if self.min_rating > 0:
            query["rating"] = {"$gte": self.min_rating}
        
        # Labels
        if self.labels:
            query["label"] = {"$in": list(self.labels)}
        
        # Text search
        if self.search_text:
            query["$or"] = [
                {"name": {"$regex": self.search_text, "$options": "i"}},
                {"description": {"$regex": self.search_text, "$options": "i"}},
            ]
        
        # Directory path
        if self.directory_path:
            if self.include_subdirs:
                query["path"] = {"$regex": f"^{self.directory_path}"}
            else:
                query["parent_path"] = self.directory_path
        
        return query


class FilterManager(QObject):
    """
    Centralized filter state manager.
    
    Emits signals when filters change so all views update consistently.
    
    Usage:
        filter_mgr = FilterManager()
        filter_mgr.filter_changed.connect(self.refresh_view)
        
        # Update filter
        filter_mgr.set_tag_filter([tag_id1, tag_id2])
        
        # Get query
        query = filter_mgr.get_mongo_query()
    """
    
    # Signals
    filter_changed = Signal()  # Emitted when any filter changes
    filters_changed = Signal()  # Alias for filter_changed (compatibility)
    tags_changed = Signal(list)  # List of selected tag IDs
    album_changed = Signal(object)  # Selected album ID or None
    search_changed = Signal(str)  # Search text
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = FilterState()
        logger.info("FilterManager initialized")
    
    @property
    def state(self) -> FilterState:
        """Get current filter state (read-only)."""
        return self._state
    
    def is_active(self) -> bool:
        """Check if any filters are active."""
        return self._state.is_active()
    
    def get_mongo_query(self) -> Dict[str, Any]:
        """Get MongoDB query for current filters."""
        return self._state.to_mongo_query()
    
    # ==================== Tag Filters ====================
    
    def set_tag_filter(
        self, 
        tag_ids: List[ObjectId],
        mode: str = "any"
    ) -> None:
        """
        Set tag filter.
        
        Args:
            tag_ids: List of tag ObjectIds
            mode: "any", "all", or "none"
        """
        self._state.selected_tag_ids = set(tag_ids)
        self._state.tag_mode = mode
        self.tags_changed.emit(list(tag_ids))
        self._emit_changed()
    
    def add_tag_filter(self, tag_id: ObjectId) -> None:
        """Add single tag to filter."""
        self._state.selected_tag_ids.add(tag_id)
        self.tags_changed.emit(list(self._state.selected_tag_ids))
        self._emit_changed()
    
    def remove_tag_filter(self, tag_id: ObjectId) -> None:
        """Remove single tag from filter."""
        self._state.selected_tag_ids.discard(tag_id)
        self.tags_changed.emit(list(self._state.selected_tag_ids))
        self._emit_changed()
    
    def clear_tag_filter(self) -> None:
        """Clear all tag filters."""
        self._state.selected_tag_ids.clear()
        self.tags_changed.emit([])
        self._emit_changed()
    
    # ==================== Album Filter ====================
    
    def set_album_filter(self, album_id: Optional[ObjectId]) -> None:
        """Set album filter."""
        self._state.selected_album_id = album_id
        self.album_changed.emit(album_id)
        self._emit_changed()
    
    def clear_album_filter(self) -> None:
        """Clear album filter."""
        self._state.selected_album_id = None
        self.album_changed.emit(None)
        self._emit_changed()
    
    # ==================== Search Filter ====================
    
    def set_search_text(self, text: str) -> None:
        """Set text search filter."""
        self._state.search_text = text.strip()
        self.search_changed.emit(self._state.search_text)
        self._emit_changed()
    
    def clear_search(self) -> None:
        """Clear search filter."""
        self._state.search_text = ""
        self.search_changed.emit("")
        self._emit_changed()
    
    # ==================== File Type Filter ====================
    
    def set_file_types(self, types: List[str]) -> None:
        """Set file type filter."""
        self._state.file_types = set(types)
        self._emit_changed()
    
    def clear_file_types(self) -> None:
        """Clear file type filter."""
        self._state.file_types.clear()
        self._emit_changed()
    
    # ==================== Rating Filter ====================
    
    def set_rating_filter(self, min_rating: int, max_rating: int = 5) -> None:
        """Set rating filter."""
        self._state.min_rating = max(0, min(5, min_rating))
        self._state.max_rating = max(0, min(5, max_rating))
        self._emit_changed()
    
    # ==================== Directory Filter ====================
    
    def set_directory_filter(
        self, 
        path: Optional[str],
        include_subdirs: bool = True
    ) -> None:
        """Set directory path filter."""
        self._state.directory_path = path
        self._state.include_subdirs = include_subdirs
        self._emit_changed()
    
    # ==================== Clear All ====================
    
    def clear_all(self) -> None:
        """Clear all filters."""
        self._state = FilterState()
        self.tags_changed.emit([])
        self.album_changed.emit(None)
        self.search_changed.emit("")
        self._emit_changed()
    
    def _emit_changed(self) -> None:
        """Emit filter_changed signal."""
        self.filter_changed.emit()
        self.filters_changed.emit()  # Compatibility
        logger.debug(f"Filter changed: active={self.is_active()}")
    
    def active_filter_count(self) -> int:
        """Get count of active filters."""
        count = 0
        if self._state.selected_tag_ids:
            count += 1
        if self._state.selected_album_id:
            count += 1
        if self._state.file_types:
            count += 1
        if self._state.labels:
            count += 1
        if self._state.search_text:
            count += 1
        if self._state.min_rating > 0:
            count += 1
        if self._state.directory_path:
            count += 1
        return count
    
    def get_query(self):
        """
        Get Q expression for current filters.
        
        Returns:
            Q expression or None if no filters active
        """
        from src.ucorefs.query.builder import Q
        
        if not self.is_active():
            return None
        
        conditions = []
        
        # Tag filtering
        if self._state.selected_tag_ids:
            tag_list = list(self._state.selected_tag_ids)
            if len(tag_list) == 1:
                conditions.append(Q.has_tag(tag_list[0]))
            else:
                # For multiple tags, depends on mode
                for tid in tag_list:
                    conditions.append(Q.has_tag(tid))
        
        # File types
        if self._state.file_types:
            types = list(self._state.file_types)
            if len(types) == 1:
                conditions.append(Q.file_type(types[0]))
            else:
                type_qs = [Q.file_type(t) for t in types]
                conditions.append(Q.OR(*type_qs))
        
        # Rating
        if self._state.min_rating > 0:
            conditions.append(Q.rating_gte(self._state.min_rating))
        
        # Text search
        if self._state.search_text:
            conditions.append(Q.name_contains(self._state.search_text))
        
        # Combine
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return Q.AND(*conditions)

