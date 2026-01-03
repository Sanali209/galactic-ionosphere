"""
UExplorer - Card View Model

ViewModel for CardGridView using QueryBuilder.
Loads all images from database with pagination.
"""
from typing import List, Optional
from bson import ObjectId
from PySide6.QtCore import QObject, Signal
from loguru import logger


class CardViewModel(QObject):
    """
    ViewModel for CardGridView.
    
    Uses QueryBuilder to load all images from database.
    Supports pagination, filtering, and both text/vector search.
    
    Signals:
        files_changed: Emitted when files list changes
        loading_changed: Emitted when loading state changes
        error_occurred: Emitted on error
    """
    
    files_changed = Signal(list)
    loading_changed = Signal(bool)
    error_occurred = Signal(str)
    page_changed = Signal(int)
    total_changed = Signal(int)
    
    def __init__(self, locator=None, parent=None):
        super().__init__(parent)
        self._locator = locator
        self._files: List = []
        self._page: int = 0
        self._page_size: int = 100
        self._total: int = 0
        self._is_loading: bool = False
        self._current_filter = None
        
        self.initialize_reactivity()
        logger.info("CardViewModel initialized")
    
    @property
    def files(self) -> List:
        return self._files
    
    @property
    def page(self) -> int:
        return self._page
    
    @property
    def total(self) -> int:
        return self._total
    
    @property
    def is_loading(self) -> bool:
        return self._is_loading
    
    def set_filter(self, filter_query):
        """Set current filter (Q expression)."""
        self._current_filter = filter_query
    
    async def load_page(self, page: int = 0):
        """
        Load a page of images from database.
        
        Args:
            page: Page number (0-indexed)
        """
        if self._is_loading:
            return
        
        self._is_loading = True
        self.loading_changed.emit(True)
        
        try:
            from src.ucorefs.query.builder import QueryBuilder, Q
            from src.ucorefs.models import FileRecord
            
            # Build query for images
            builder = QueryBuilder()
            builder = builder.where(Q.file_type("image"))
            
            # Add current filter if set
            if self._current_filter:
                builder = builder.where(self._current_filter)
            
            # Pagination
            skip = page * self._page_size
            builder = builder.limit(self._page_size).skip(skip)
            
            # Execute query
            self._files = await builder.execute()
            self._page = page
            
            # Get total count
            self._total = await FileRecord.count(builder.get_query())
            
            self.files_changed.emit(self._files)
            self.page_changed.emit(self._page)
            self.total_changed.emit(self._total)
            
            logger.debug(f"Loaded page {page}: {len(self._files)} files")
            
        except Exception as e:
            logger.error(f"Failed to load page: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._is_loading = False
            self.loading_changed.emit(False)
    
    async def load_next_page(self):
        """Load next page."""
        await self.load_page(self._page + 1)
    
    async def load_previous_page(self):
        """Load previous page."""
        if self._page > 0:
            await self.load_page(self._page - 1)
    
    async def refresh(self):
        """Refresh current page."""
        await self.load_page(self._page)
    
    async def search_text(self, text: str):
        """
        Search by text in file names.
        
        Args:
            text: Search text
        """
        from src.ucorefs.query.builder import Q
        
        if text:
            self.set_filter(Q.name_contains(text))
        else:
            self.set_filter(None)
        
        await self.load_page(0)
    
    async def search_vector(self, query_text: str, threshold: float = 0.7):
        """
        Search by vector similarity.
        
        Args:
            query_text: Text to convert to embedding
            threshold: Similarity threshold
        """
        try:
            from src.ucorefs.vectors.service import VectorService
            from src.ucorefs.query.builder import QueryBuilder
            
            if not self._locator:
                self.error_occurred.emit("No locator available")
                return
            
            vector_service = self._locator.get_system(VectorService)
            if not vector_service or not vector_service.is_available():
                self.error_occurred.emit("Vector service not available")
                return
            
            # Get embedding for search text
            embedding = await vector_service.get_text_embedding(query_text)
            if not embedding:
                self.error_occurred.emit("Could not generate embedding")
                return
            
            # Build query with vector similarity
            builder = QueryBuilder()
            builder = builder.vector_similar(embedding, threshold=threshold)
            
            # Add current filter
            if self._current_filter:
                builder = builder.where(self._current_filter)
            
            # Execute
            self._files = await builder.execute(vector_service)
            self._page = 0
            self._total = len(self._files)
            
            self.files_changed.emit(self._files)
            self.total_changed.emit(self._total)
            
            logger.info(f"Vector search found {len(self._files)} similar files")
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self.error_occurred.emit(str(e))

    # === Reactive SSOT Implementation ===

    @property
    def _event_bus(self):
        """Lazy access to EventBus."""
        from src.core.events import EventBus
        try:
            return self._locator.get_system(EventBus)
        except Exception:
            return None

    def initialize_reactivity(self):
        """Subscribe to database events."""
        bus = self._event_bus
        if bus:
             bus.subscribe("db.file_records.updated", self._on_file_updated)
             bus.subscribe("db.file_records.deleted", self._on_file_deleted)
             logger.debug("CardViewModel: Reactivity initialized")

    def shutdown(self):
        """Cleanup subscriptions."""
        bus = self._event_bus
        if bus:
            bus.unsubscribe("db.file_records.updated", self._on_file_updated)
            bus.unsubscribe("db.file_records.deleted", self._on_file_deleted)

    def _on_file_updated(self, data: dict):
        """Handle real-time file updates."""
        try:
            file_id = data.get("id")
            if not file_id: return

            updated = False
            for record in self._files:
                if getattr(record, "_id", None) == file_id:
                    # Update fields
                    record_data = data.get("record", {})
                    for k, v in record_data.items():
                         if hasattr(record, k) and k != "_id":
                             setattr(record, k, v)
                    updated = True
            
            if updated:
                self.files_changed.emit(self._files)
                logger.debug(f"CardViewModel: File {file_id} refreshed")
                
        except Exception as e:
             logger.error(f"Error handling file update: {e}")

    def _on_file_deleted(self, data: dict):
        """Handle real-time file deletion."""
        try:
            file_id = data.get("id")
            if not file_id: return
            
            initial_len = len(self._files)
            self._files = [f for f in self._files if getattr(f, "_id", None) != file_id]
            
            if len(self._files) != initial_len:
                self.files_changed.emit(self._files)
                logger.debug(f"CardViewModel: File {file_id} removed")
        except Exception as e:
            logger.error(f"Error handling file deletion: {e}")
