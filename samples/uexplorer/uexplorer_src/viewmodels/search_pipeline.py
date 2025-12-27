"""
SearchPipeline - Executes searches with 3 modes.

Modes:
- text: MongoDB regex search on selected fields
- vector: Text → Embedding → FAISS similarity
- image: Image → Embedding → FAISS similarity
"""
from typing import List, Optional
from PySide6.QtCore import QObject, Signal
from bson import ObjectId
from loguru import logger

from uexplorer_src.viewmodels.search_query import SearchQuery
from src.ucorefs.query import Q, F


class SearchPipeline(QObject):
    """
    Executes searches and emits results.
    
    Supports text, vector, and image search modes.
    Combines with filters and tags into unified query.
    
    Signals:
        search_started: Search began
        search_completed: Results ready
        search_failed: Error occurred
    """
    
    search_started = Signal()
    search_completed = Signal(list)  # List[FileRecord]
    search_failed = Signal(str)  # Error message
    
    def __init__(self, locator, parent=None):
        """
        Initialize pipeline.
        
        Args:
            locator: ServiceLocator for accessing services
        """
        super().__init__(parent)
        self._locator = locator
    
    async def execute(self, query: SearchQuery) -> List:
        """
        Execute search query.
        
        Args:
            query: SearchQuery with all parameters
            
        Returns:
            List of FileRecord objects
        """
        self.search_started.emit()
        
        try:
            if query.is_image_search():
                results = await self._image_search(query)
            elif query.is_vector_search():
                results = await self._vector_search(query)
            else:
                results = await self._text_search(query)
            
            self.search_completed.emit(results)
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Search failed: {error_msg}")
            self.search_failed.emit(error_msg)
            return []
    
    async def _text_search(self, query: SearchQuery) -> List:
        """
        Text search using MongoDB regex.
        
        Searches selected fields with case-insensitive regex.
        """
        from src.ucorefs.models import FileRecord
        
        conditions = []
        
        # Build text search condition
        if query.text:
            field_conditions = []
            for field in query.fields:
                if field == "name":
                    field_conditions.append(Q.name_contains(query.text))
                elif field == "path":
                    field_conditions.append(F("path").regex(query.text, "i"))
                elif field == "description":
                    field_conditions.append(F("description").regex(query.text, "i"))
                elif field == "ai_description":
                    field_conditions.append(F("ai_description").regex(query.text, "i"))
            
            if field_conditions:
                conditions.append(Q.OR(*field_conditions))
        
        # Add filter conditions
        conditions.extend(self._build_filter_conditions(query.filters))
        
        # Add tag conditions
        if query.tags:
            conditions.append(self._build_tag_condition(query.tags, query.tag_mode))
        
        # Add directory scope
        if query.directory:
            conditions.append(F("parent_id").eq(query.directory))
        
        # Build final query
        if conditions:
            mongo_query = Q.AND(*conditions).query
        else:
            mongo_query = {}
        
        # Execute
        results = await FileRecord.find(mongo_query, limit=query.limit)
        logger.info(f"Text search found {len(results)} results")
        return results
    
    async def _vector_search(self, query: SearchQuery) -> List:
        """
        Vector search using unified SearchService.
        
        Uses SearchService for combined text→embedding→FAISS similarity.
        This demonstrates proper use of Foundation's unified search API.
        """
        from src.ucorefs.search.service import SearchService, SearchQuery as SvcSearchQuery
        from src.ucorefs.models import FileRecord
        
        try:
            search_service = self._locator.get_system(SearchService)
        except KeyError:
            logger.warning("SearchService not available, falling back to VectorService")
            return await self._vector_search_fallback(query)
        
        try:
            # Build SearchQuery using Foundation's SearchService dataclass
            svc_query = SvcSearchQuery(
                text=query.text,
                vector_search=True,
                vector_provider="clip",
                filters=query.filters,
                limit=query.limit
            )
            
            # Execute unified search
            results = await search_service.search(svc_query)
            
            if not results:
                return []
            
            # Get FileRecords from result IDs
            file_ids = [r.file_id for r in results]
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            
            logger.info(f"Vector search (via SearchService) found {len(files)} results")
            return files
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _vector_search_fallback(self, query: SearchQuery) -> List:
        """Fallback to direct VectorService if SearchService unavailable."""
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.models import FileRecord
        
        vector_service = self._locator.get_system(VectorService)
        if not vector_service:
            return []
        
        try:
            results = await vector_service.search_by_text(query.text, limit=query.limit)
            if not results:
                return []
            
            file_ids = [ObjectId(r["file_id"]) for r in results if "file_id" in r]
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            return files
        except Exception as e:
            logger.error(f"Vector fallback failed: {e}")
            return []
    
    async def _image_search(self, query: SearchQuery) -> List:
        """
        Image similarity search.
        
        Gets embedding from image and finds similar via FAISS.
        """
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.models import FileRecord
        
        vector_service = self._locator.get_system(VectorService)
        if not vector_service:
            logger.warning("VectorService not available")
            return []
        
        try:
            # Find similar to image
            results = await vector_service.find_similar(
                file_id=str(query.file_id),
                limit=query.limit
            )
            
            if not results:
                return []
            
            # Convert to FileRecords
            file_ids = [ObjectId(r["file_id"]) for r in results if "file_id" in r]
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            
            logger.info(f"Image search found {len(files)} similar")
            return files
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    def _build_filter_conditions(self, filters: dict) -> List[Q]:
        """Build Q conditions from filter dict including include/exclude."""
        conditions = []
        
        for field, value in filters.items():
            if not value:
                continue
                
            if field == "file_type":
                if isinstance(value, list):
                    conditions.append(F("file_type").in_(value))
                else:
                    conditions.append(F("file_type").eq(value))
            
            elif field == "rating" and value is not None:
                conditions.append(F("rating").gte(value))
            
            elif field == "extension":
                if isinstance(value, list):
                    conditions.append(F("extension").in_(value))
                else:
                    conditions.append(F("extension").eq(value))
            
            # ===== NEW: Include/Exclude Tags =====
            elif field == "tag_include":
                # All included tags must be present
                tag_ids = [ObjectId(t) if isinstance(t, str) else t for t in value]
                conditions.append(F("tag_ids").all_(tag_ids))
            
            elif field == "tag_exclude":
                # None of excluded tags should be present
                tag_ids = [ObjectId(t) if isinstance(t, str) else t for t in value]
                conditions.append(F("tag_ids").nin(tag_ids))
            
            # ===== NEW: Include/Exclude Albums =====
            elif field == "album_include":
                # All included albums must contain the file
                album_ids = [ObjectId(a) if isinstance(a, str) else a for a in value]
                conditions.append(F("album_ids").all_(album_ids))
            
            elif field == "album_exclude":
                # File should not be in excluded albums
                album_ids = [ObjectId(a) if isinstance(a, str) else a for a in value]
                conditions.append(F("album_ids").nin(album_ids))
            
            # ===== NEW: Include/Exclude Directories (by path prefix) =====
            elif field == "directory_include":
                # Files must be in one of the included directory paths
                # Use $or with path regex for prefix matching
                path_conditions = [F("path").regex(f"^{p}") for p in value]
                if path_conditions:
                    conditions.append(Q.OR(*path_conditions))
            
            elif field == "directory_exclude":
                # Files must not be in any excluded directory paths
                for path in value:
                    conditions.append(~F("path").regex(f"^{path}"))
        
        return conditions
    
    def _build_tag_condition(self, tags: List[ObjectId], mode: str) -> Q:
        """Build tag filter condition (legacy support)."""
        if mode == "any":
            return F("tag_ids").in_(tags)
        elif mode == "all":
            return F("tag_ids").all_(tags)
        elif mode == "none":
            return F("tag_ids").nin(tags)
        return Q({})

