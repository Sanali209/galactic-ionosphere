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
from src.ucorefs.query.builder import Q


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
                    field_conditions.append(Q({
                        "path": {"$regex": query.text, "$options": "i"}
                    }))
                elif field == "description":
                    field_conditions.append(Q({
                        "description": {"$regex": query.text, "$options": "i"}
                    }))
                elif field == "ai_description":
                    field_conditions.append(Q({
                        "ai_description": {"$regex": query.text, "$options": "i"}
                    }))
            
            if field_conditions:
                conditions.append(Q.OR(*field_conditions))
        
        # Add filter conditions
        conditions.extend(self._build_filter_conditions(query.filters))
        
        # Add tag conditions
        if query.tags:
            conditions.append(self._build_tag_condition(query.tags, query.tag_mode))
        
        # Add directory scope
        if query.directory:
            conditions.append(Q({"parent_id": query.directory}))
        
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
        Vector search using text → embedding → FAISS.
        
        Converts query text to embedding and finds similar images.
        """
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.models import FileRecord
        
        vector_service = self._locator.get_system(VectorService)
        if not vector_service:
            logger.warning("VectorService not available")
            return []
        
        try:
            # Text to embedding search
            results = await vector_service.search_by_text(
                query.text, 
                limit=query.limit
            )
            
            if not results:
                return []
            
            # Convert to FileRecords
            file_ids = [ObjectId(r["file_id"]) for r in results if "file_id" in r]
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            
            logger.info(f"Vector search found {len(files)} results")
            return files
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
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
        """Build Q conditions from filter dict."""
        conditions = []
        
        for field, value in filters.items():
            if field == "file_type" and value:
                if isinstance(value, list):
                    conditions.append(Q({"file_type": {"$in": value}}))
                else:
                    conditions.append(Q.file_type(value))
            
            elif field == "rating" and value is not None:
                conditions.append(Q.rating_gte(value))
            
            elif field == "extension" and value:
                if isinstance(value, list):
                    conditions.append(Q.extension_in(value))
                else:
                    conditions.append(Q({"extension": value}))
        
        return conditions
    
    def _build_tag_condition(self, tags: List[ObjectId], mode: str) -> Q:
        """Build tag filter condition."""
        if mode == "any":
            return Q({"tag_ids": {"$in": tags}})
        elif mode == "all":
            return Q({"tag_ids": {"$all": tags}})
        elif mode == "none":
            return Q({"tag_ids": {"$nin": tags}})
        return Q({})
