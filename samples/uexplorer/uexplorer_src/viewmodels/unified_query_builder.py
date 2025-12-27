"""
UnifiedQueryBuilder - Collects search criteria from all panels.

Central component that:
1. Connects to all filter source panels (Search, Tags, Albums, Directories, Filters)
2. Builds unified query combining all criteria
3. Emits query_changed signal when any source changes

Usage:
    builder = UnifiedQueryBuilder(locator)
    builder.connect_panels(search_panel, tag_panel, album_panel, filter_panel)
    builder.query_changed.connect(execute_search)
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal
from bson import ObjectId
from loguru import logger


@dataclass
class UnifiedSearchQuery:
    """
    Unified query combining all search criteria.
    
    Attributes:
        mode: "text", "semantic", or "similar"
        text: Text query string
        text_fields: Fields to search for text mode
        similar_file_id: File ID for "similar" mode
        
        tag_include: Tags that must be present
        tag_exclude: Tags that must not be present
        album_include: Albums to include
        album_exclude: Albums to exclude
        directory_include: Directory paths to include
        directory_exclude: Directory paths to exclude
        
        filters: Dict of field filters from FilterPanel
        limit: Max results
    """
    # Search mode
    mode: str = "text"  # "text", "semantic", "similar"
    
    # Text/Semantic search
    text: str = ""
    text_fields: List[str] = field(default_factory=lambda: ["name", "path"])
    
    # Similar search (Image → Vector)
    similar_file_id: Optional[ObjectId] = None
    
    # Tag filters
    tag_include: List[str] = field(default_factory=list)
    tag_exclude: List[str] = field(default_factory=list)
    
    # Album filters
    album_include: List[str] = field(default_factory=list)
    album_exclude: List[str] = field(default_factory=list)
    
    # Directory filters
    directory_include: List[str] = field(default_factory=list)
    directory_exclude: List[str] = field(default_factory=list)
    
    # Field filters from FilterPanel
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.tag_include or self.tag_exclude or
            self.album_include or self.album_exclude or
            self.directory_include or self.directory_exclude or
            self.filters
        )
    
    def to_mongo_filter(self) -> Dict[str, Any]:
        """Build MongoDB filter dict from query."""
        mongo = {}
        
        # Tag filters
        if self.tag_include:
            # Files must have ALL included tags
            mongo["tag_ids"] = {"$all": [ObjectId(t) for t in self.tag_include]}
        
        if self.tag_exclude:
            # Files must NOT have ANY excluded tags
            # Use $nor to properly exclude files containing any excluded tag
            if "$nor" not in mongo:
                mongo["$nor"] = []
            for tag_id in self.tag_exclude:
                mongo["$nor"].append({"tag_ids": ObjectId(tag_id)})
        
        # Album filters
        if self.album_include:
            # Files must have ALL included albums
            mongo["album_ids"] = {"$all": [ObjectId(a) for a in self.album_include]}
        
        if self.album_exclude:
            # Files must NOT have ANY excluded albums
            # Use $nor to properly exclude files containing any excluded album
            if "$nor" not in mongo:
                mongo["$nor"] = []
            for album_id in self.album_exclude:
                mongo["$nor"].append({"album_ids": ObjectId(album_id)})
        
        # Directory filters (path prefix matching)
        if self.directory_include:
            # Files must be in one of the included directories
            mongo["$or"] = [{"path": {"$regex": f"^{d}"}} for d in self.directory_include]
        if self.directory_exclude:
            # Files must not be in excluded directories
            for d in self.directory_exclude:
                if "$nor" not in mongo:
                    mongo["$nor"] = []
                mongo["$nor"].append({"path": {"$regex": f"^{d}"}})
        
        # Field filters from FilterPanel
        for field_name, value in self.filters.items():
            if value is not None and value != "":
                # Special handling for unrated filter
                if field_name == "unrated" and value is True:
                    # Show files with no rating (null, 0, or missing field)
                    mongo["$or"] = [
                        {"rating": {"$exists": False}},  # No rating field
                        {"rating": None},                 # rating is null
                        {"rating": 0}                     # rating is 0
                    ]
                # Special handling for rating - use >= for "X stars or better"
                elif field_name == "rating":
                    mongo[field_name] = {"$gte": value}
                elif field_name != "unrated":  # Skip unrated flag itself
                    mongo[field_name] = value
        
        # Debug logging to see what query is generated
        if mongo:
            logger.debug(f"MongoDB filter generated: {mongo}")
        
        return mongo


class UnifiedQueryBuilder(QObject):
    """
    Builds unified query from all filter sources.
    
    Signals:
        query_changed(UnifiedSearchQuery): Emitted when any filter changes
    """
    
    query_changed = Signal(object)  # UnifiedSearchQuery
    
    def __init__(self, locator=None, parent=None):
        super().__init__(parent)
        self._locator = locator
        
        # Panel references
        self._search_panel = None
        self._tag_panel = None
        self._album_panel = None
        self._directory_panel = None
        self._filter_panel = None
        
        # Current query state
        self._current_query = UnifiedSearchQuery()
        
        logger.debug("UnifiedQueryBuilder initialized")
    
    def connect_search_panel(self, panel):
        """Connect to SearchDockPanel."""
        self._search_panel = panel
        if hasattr(panel, 'search_requested'):
            panel.search_requested.connect(self._on_search_requested)
        logger.debug("Connected to SearchPanel")
    
    def connect_tag_panel(self, panel):
        """Connect to TagPanel."""
        self._tag_panel = panel
        if hasattr(panel, 'filter_changed'):
            panel.filter_changed.connect(self._on_tag_filter_changed)
        logger.debug("Connected to TagPanel")
    
    def connect_album_panel(self, panel):
        """Connect to AlbumPanel."""
        self._album_panel = panel
        if hasattr(panel, 'filter_changed'):
            panel.filter_changed.connect(self._on_album_filter_changed)
        logger.debug("Connected to AlbumPanel")
    
    def connect_directory_panel(self, panel):
        """Connect to DirectoryPanel."""
        self._directory_panel = panel
        if hasattr(panel, 'filter_changed'):
            panel.filter_changed.connect(self._on_directory_filter_changed)
        logger.debug("Connected to DirectoryPanel")
    
    def connect_filter_panel(self, panel):
        """Connect to FilterPanel."""
        self._filter_panel = panel
        if hasattr(panel, 'filters_changed'):
            panel.filters_changed.connect(self._on_filters_changed)
        logger.debug("Connected to FilterPanel")
    
    def connect_all_panels(
        self,
        search_panel=None,
        tag_panel=None,
        album_panel=None,
        directory_panel=None,
        filter_panel=None
    ):
        """Connect all panels at once."""
        if search_panel:
            self.connect_search_panel(search_panel)
        if tag_panel:
            self.connect_tag_panel(tag_panel)
        if album_panel:
            self.connect_album_panel(album_panel)
        if directory_panel:
            self.connect_directory_panel(directory_panel)
        if filter_panel:
            self.connect_filter_panel(filter_panel)
    
    def _on_search_requested(self, mode: str, query: str, fields: list):
        """Handle search request from SearchPanel."""
        self._current_query.mode = mode
        self._current_query.text = query
        self._current_query.text_fields = fields
        self._emit_query()
    
    def _on_tag_filter_changed(self, include: list, exclude: list):
        """Handle tag filter change."""
        self._current_query.tag_include = include
        self._current_query.tag_exclude = exclude
        self._emit_query()
    
    def _on_album_filter_changed(self, include: list, exclude: list):
        """Handle album filter change."""
        self._current_query.album_include = include
        self._current_query.album_exclude = exclude
        self._emit_query()
    
    def _on_directory_filter_changed(self, include: list, exclude: list):
        """Handle directory filter change."""
        self._current_query.directory_include = include
        self._current_query.directory_exclude = exclude
        self._emit_query()
    
    def _on_filters_changed(self, filters: dict):
        """Handle field filters change from FilterPanel."""
        self._current_query.filters = filters
        self._emit_query()
    
    def set_similar_file(self, file_id: ObjectId):
        """Set file for similar search (from context menu)."""
        self._current_query.mode = "similar"
        self._current_query.similar_file_id = file_id
        self._emit_query()
    
    def get_current_query(self) -> UnifiedSearchQuery:
        """Get current query state."""
        return self._current_query
    
    def set_text_search(self, mode: str, text: str, fields: list):
        """Set text search parameters from UnifiedSearchPanel."""
        self._current_query.mode = mode
        self._current_query.text = text
        self._current_query.text_fields = fields
        self._emit_query()
    
    def set_filters(self, filters: dict):
        """Set field filters from UnifiedSearchPanel."""
        # Merge with existing filters
        self._current_query.filters.update(filters)
        self._emit_query()
    
    def _emit_query(self):
        """Emit query changed signal."""
        logger.debug(f"Query changed: mode={self._current_query.mode}, "
                    f"text='{self._current_query.text[:20] if self._current_query.text else ''}', "
                    f"tags=+{len(self._current_query.tag_include)}/-{len(self._current_query.tag_exclude)}")
        self.query_changed.emit(self._current_query)
    
    def clear_all(self):
        """Clear all query state and notify connected panels."""
        self._current_query = UnifiedSearchQuery()
        
        # Clear connected panels
        if self._tag_panel and hasattr(self._tag_panel, 'clear_filters'):
            self._tag_panel.clear_filters()
        if self._album_panel and hasattr(self._album_panel, 'clear_filters'):
            self._album_panel.clear_filters()
        if self._directory_panel and hasattr(self._directory_panel, 'clear_filters'):
            self._directory_panel.clear_filters()
        
        self._emit_query()

