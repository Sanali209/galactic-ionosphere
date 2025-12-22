"""
BrowseViewModel - Per-document state manager.

Each file browser document has its own ViewModel that:
- Tracks current directory, view mode, selection
- Receives and stores search results
- Emits signals when state changes
"""
from typing import List, Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
from bson import ObjectId
from loguru import logger


class BrowseViewModel(QObject):
    """
    Per-document ViewModel for file browsing.
    
    Signals:
        results_changed: FileRecord list changed
        loading_changed: Loading state changed
        view_mode_changed: View mode (tree/list/card) changed
        selection_changed: Selected items changed
        directory_changed: Current directory changed
    """
    
    # State change signals
    results_changed = Signal(list)  # List[FileRecord]
    loading_changed = Signal(bool)
    view_mode_changed = Signal(str)  # "tree" | "list" | "card"
    selection_changed = Signal(list)  # List[ObjectId]
    directory_changed = Signal(object)  # ObjectId or None
    sort_changed = Signal(str, bool)  # field, ascending
    group_changed = Signal(str)  # group_by field or None
    error_occurred = Signal(str)
    
    def __init__(self, doc_id: str, parent=None):
        """
        Initialize ViewModel for a document.
        
        Args:
            doc_id: Unique document identifier
        """
        super().__init__(parent)
        self._doc_id = doc_id
        
        # State
        self._results: List[Any] = []  # FileRecords - EMPTY on init
        self._is_loading: bool = False
        self._view_mode: str = "tree"  # tree | list | card
        self._selection: List[ObjectId] = []
        self._current_directory: Optional[ObjectId] = None
        
        # Sort/Group state
        self._sort_field: str = "name"
        self._sort_ascending: bool = True
        self._group_by: Optional[str] = None
        
        # Query state (what filters are active)
        self._search_text: str = ""
        self._search_mode: str = "text"  # text | vector | image
        self._search_fields: List[str] = ["name"]
        self._active_filters: Dict[str, Any] = {}
        self._selected_tags: List[ObjectId] = []
        self._tag_mode: str = "any"  # any | all | none
        
        logger.debug(f"BrowseViewModel created: {doc_id}")
    
    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self._doc_id
    
    @property
    def results(self) -> List[Any]:
        """Get current results."""
        return self._results
    
    @property
    def is_loading(self) -> bool:
        """Check if loading."""
        return self._is_loading
    
    @property
    def view_mode(self) -> str:
        """Get current view mode."""
        return self._view_mode
    
    @property
    def selection(self) -> List[ObjectId]:
        """Get selected item IDs."""
        return self._selection
    
    @property
    def current_directory(self) -> Optional[ObjectId]:
        """Get current directory."""
        return self._current_directory
    
    @property
    def sort_field(self) -> str:
        """Get current sort field."""
        return self._sort_field
    
    @property
    def sort_ascending(self) -> bool:
        """Get sort direction."""
        return self._sort_ascending
    
    @property
    def group_by(self) -> Optional[str]:
        """Get grouping field."""
        return self._group_by
    
    # === State Setters ===
    
    def set_results(self, results: List[Any]):
        """
        Set search results.
        
        Args:
            results: List of FileRecord objects
        """
        self._results = results
        self.results_changed.emit(results)
        logger.info(f"[{self._doc_id}] Results updated: {len(results)} items")
    
    def set_loading(self, loading: bool):
        """Set loading state."""
        if self._is_loading != loading:
            self._is_loading = loading
            self.loading_changed.emit(loading)
    
    def set_view_mode(self, mode: str):
        """
        Set view mode (tree/list/card).
        
        Results persist across mode changes.
        """
        if mode not in ("tree", "list", "card"):
            logger.warning(f"Invalid view mode: {mode}")
            return
        
        if self._view_mode != mode:
            self._view_mode = mode
            self.view_mode_changed.emit(mode)
            logger.debug(f"[{self._doc_id}] View mode: {mode}")
    
    def set_selection(self, selection: List[ObjectId]):
        """Set selected items."""
        if self._selection != selection:
            self._selection = selection
            self.selection_changed.emit(selection)
    
    def set_sort(self, field: str, ascending: bool = True):
        """
        Set sort field and direction.
        
        Args:
            field: Field to sort by (name, size, modified, rating)
            ascending: True for ascending, False for descending
        """
        if self._sort_field != field or self._sort_ascending != ascending:
            self._sort_field = field
            self._sort_ascending = ascending
            self.sort_changed.emit(field, ascending)
            logger.debug(f"[{self._doc_id}] Sort: {field} {'asc' if ascending else 'desc'}")
    
    def set_group(self, group_by: Optional[str] = None):
        """
        Set grouping field.
        
        Args:
            group_by: Field to group by (None, file_type, date, rating)
        """
        if self._group_by != group_by:
            self._group_by = group_by
            self.group_changed.emit(group_by)
            logger.debug(f"[{self._doc_id}] Group by: {group_by}")
    
    def set_directory(self, directory_id: Optional[ObjectId]):
        """Set current directory."""
        if self._current_directory != directory_id:
            self._current_directory = directory_id
            self.directory_changed.emit(directory_id)
    
    # === Query State ===
    
    def set_search(self, text: str, mode: str = "text", fields: List[str] = None):
        """
        Set search parameters.
        
        Args:
            text: Search query text
            mode: "text", "vector", or "image"
            fields: Fields to search (for text mode)
        """
        self._search_text = text
        self._search_mode = mode
        if fields:
            self._search_fields = fields
    
    def set_filters(self, filters: Dict[str, Any]):
        """Set active filters from FilterPanel."""
        self._active_filters = filters
    
    def set_tags(self, tag_ids: List[ObjectId], mode: str = "any"):
        """
        Set tag filter.
        
        Args:
            tag_ids: Selected tag IDs
            mode: "any", "all", or "none"
        """
        self._selected_tags = tag_ids
        self._tag_mode = mode
    
    def clear_search(self):
        """Clear search and show directory contents."""
        self._search_text = ""
        self._results = []
        self.results_changed.emit([])
    
    def get_query_state(self) -> Dict[str, Any]:
        """
        Get current query state for SearchPipeline.
        
        Returns:
            Dict with all query parameters
        """
        return {
            "search_text": self._search_text,
            "search_mode": self._search_mode,
            "search_fields": self._search_fields,
            "filters": self._active_filters,
            "tags": self._selected_tags,
            "tag_mode": self._tag_mode,
            "directory": self._current_directory,
        }
    
    # === ViewSettings Sync ===
    
    async def load_view_settings(self, context_id: str = None):
        """
        Load and apply ViewSettings from database.
        
        Args:
            context_id: Context identifier (defaults to doc_id)
        """
        try:
            from src.ucorefs.models.view_settings import ViewSettingsService
            
            ctx = context_id or self._doc_id
            settings = await ViewSettingsService.get_or_create(ctx)
            
            # Apply settings without emitting (bulk update)
            self._view_mode = settings.view_mode
            self._sort_field = settings.sort_field
            self._sort_ascending = settings.sort_ascending
            self._group_by = settings.group_by
            
            # Emit signals
            self.view_mode_changed.emit(self._view_mode)
            self.sort_changed.emit(self._sort_field, self._sort_ascending)
            self.group_changed.emit(self._group_by)
            
            logger.debug(f"[{self._doc_id}] ViewSettings loaded: {settings.view_mode}")
            
        except Exception as e:
            logger.error(f"Failed to load ViewSettings: {e}")
    
    async def save_view_settings(self, context_id: str = None):
        """
        Save current settings to database.
        
        Args:
            context_id: Context identifier (defaults to doc_id)
        """
        try:
            from src.ucorefs.models.view_settings import ViewSettingsService
            
            ctx = context_id or self._doc_id
            
            # Update each setting
            await ViewSettingsService.update_setting(ctx, "view_mode", self._view_mode)
            await ViewSettingsService.update_setting(ctx, "sort_field", self._sort_field)
            await ViewSettingsService.update_setting(ctx, "sort_ascending", self._sort_ascending)
            await ViewSettingsService.update_setting(ctx, "group_by", self._group_by)
            
            logger.debug(f"[{self._doc_id}] ViewSettings saved")
            
        except Exception as e:
            logger.error(f"Failed to save ViewSettings: {e}")
