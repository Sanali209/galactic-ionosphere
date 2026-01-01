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
        
        # Enhanced logging for search mode detection
        logger.info(f"========== SEARCH EXECUTION START ==========")
        logger.info(f"Query Mode: {query.mode}")
        logger.info(f"Query Text: '{query.text}'")
        logger.info(f"Is Image Search: {query.is_image_search()}")
        logger.info(f"Is Vector Search: {query.is_vector_search()}")
        logger.info(f"Is Text Search: {query.is_text_search()}")
        logger.info(f"Filters: {query.filters}")
        logger.info(f"Limit: {query.limit}")
        
        try:
            # Check for similar mode FIRST (uses image embeddings for similarity)
            if query.mode == "similar" and query.file_id:
                logger.info(f">>> Routing to SIMILAR IMAGE SEARCH (file_id={query.file_id})")
                results = await self._image_search_similar(query)
            elif query.is_image_search():
                logger.info(">>> Routing to IMAGE SEARCH")
                results = await self._image_search(query)
            elif query.is_vector_search():
                logger.info(">>> Routing to VECTOR/SEMANTIC SEARCH")
                results = await self._vector_search(query)
            else:
                logger.info(">>> Routing to TEXT SEARCH")
                results = await self._text_search(query)
            
            logger.info(f"========== SEARCH COMPLETE: {len(results)} results ==========")
            self.search_completed.emit(results)
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"========== SEARCH FAILED: {error_msg} ==========")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        logger.info(f"[VECTOR_SEARCH] Starting semantic/vector search")
        logger.info(f"[VECTOR_SEARCH] Query text: '{query.text}'")
        logger.info(f"[VECTOR_SEARCH] Filters: {query.filters}")
        
        try:
            search_service = self._locator.get_system(SearchService)
            logger.info(f"[VECTOR_SEARCH] ✓ SearchService available")
        except KeyError as e:
            logger.warning(f"[VECTOR_SEARCH] ✗ SearchService not available: {e}")
            logger.warning(f"[VECTOR_SEARCH] Falling back to VectorService")
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
            
            logger.info(f"[VECTOR_SEARCH] Built SearchService query:")
            logger.info(f"[VECTOR_SEARCH]   - text: '{svc_query.text}'")
            logger.info(f"[VECTOR_SEARCH]   - vector_search: {svc_query.vector_search}")
            logger.info(f"[VECTOR_SEARCH]   - vector_provider: {svc_query.vector_provider}")
            logger.info(f"[VECTOR_SEARCH]   - limit: {svc_query.limit}")
            
            # Execute unified search
            logger.info(f"[VECTOR_SEARCH] Executing SearchService.search()...")
            results = await search_service.search(svc_query)
            
            logger.info(f"[VECTOR_SEARCH] SearchService returned {len(results)} SearchResults")
            
            if results:
                # Log sample results with scores
                for i, r in enumerate(results[:3]):
                    logger.info(f"[VECTOR_SEARCH] Result {i+1}: file_id={r.file_id}, score={r.score:.4f}, "
                               f"match_type={r.match_type}, vector_score={r.vector_score}, text_score={r.text_score}")
            
            if not results:
                logger.warning(f"[VECTOR_SEARCH] No results returned from SearchService")
                return []
            
            # Get FileRecords from result IDs
            file_ids = [r.file_id for r in results]
            logger.info(f"[VECTOR_SEARCH] Retrieving {len(file_ids)} FileRecords from database")
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            
            logger.info(f"[VECTOR_SEARCH] ✓ Vector search complete: {len(files)} FileRecords retrieved")
            return files
            
        except Exception as e:
            logger.error(f"[VECTOR_SEARCH] ✗ Vector search failed: {e}")
            import traceback
            logger.error(f"[VECTOR_SEARCH] Traceback: {traceback.format_exc()}")
            return []
    
    async def _vector_search_fallback(self, query: SearchQuery) -> List:
        """Fallback to direct VectorService if SearchService unavailable."""
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.models import FileRecord
        
        logger.info(f"[VECTOR_FALLBACK] Attempting VectorService fallback")
        
        try:
            vector_service = self._locator.get_system(VectorService)
            logger.info(f"[VECTOR_FALLBACK] ✓ VectorService available")
        except KeyError:
            logger.error(f"[VECTOR_FALLBACK] ✗ VectorService not available either")
            return []
        
        if not vector_service:
            logger.error(f"[VECTOR_FALLBACK] ✗ VectorService is None")
            return []
        
        try:
            logger.info(f"[VECTOR_FALLBACK] Calling VectorService.search_by_text('{query.text}', limit={query.limit})")
            results = await vector_service.search_by_text(query.text, limit=query.limit)
            
            logger.info(f"[VECTOR_FALLBACK] VectorService returned {len(results) if results else 0} results")
            
            if not results:
                logger.warning(f"[VECTOR_FALLBACK] No results from VectorService")
                return []
            
            file_ids = [ObjectId(r["file_id"]) for r in results if "file_id" in r]
            logger.info(f"[VECTOR_FALLBACK] Extracting {len(file_ids)} file IDs")
            
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            logger.info(f"[VECTOR_FALLBACK] ✓ Fallback complete: {len(files)} FileRecords retrieved")
            return files
        except Exception as e:
            logger.error(f"[VECTOR_FALLBACK] ✗ Fallback failed: {e}")
            import traceback
            logger.error(f"[VECTOR_FALLBACK] Traceback: {traceback.format_exc()}")
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
    
    async def _image_search_similar(self, query: SearchQuery) -> List:
        """
        Find similar images using CLIP embeddings from FileRecord.
        
        Used by Find Similar context menu - gets source image embedding
        and searches FAISS index for visually similar images.
        """
        from src.ucorefs.vectors.faiss_service import FAISSIndexService
        from src.ucorefs.models import FileRecord
        import numpy as np
        
        logger.info(f"[SIMILAR_SEARCH] Finding images similar to file_id={query.file_id}")
        
        try:
            #Get source image FileRecord to extract CLIP embedding
            source_file = await FileRecord.get(query.file_id)
            if not source_file:
                logger.warning(f"[SIMILAR_SEARCH] Source file not found: {query.file_id}")
                return []
            
            # Extract CLIP embedding from FileRecord.embeddings
            if not hasattr(source_file, 'embeddings') or not source_file.embeddings:
                logger.warning(f"[SIMILAR_SEARCH] Source file has no embeddings")
                return []
            
            clip_data = source_file.embeddings.get('clip')
            if not clip_data or 'vector' not in clip_data:
                logger.warning(f"[SIMILAR_SEARCH] Source file has no CLIP vector")
                return []
            
            source_vector = np.array(clip_data['vector'], dtype=np.float32)
            logger.info(f"[SIMILAR_SEARCH] Loaded CLIP vector: shape={source_vector.shape}")
            
            # Use FAISS to find similar images
            faiss_service = self._locator.get_system(FAISSIndexService)
            if not faiss_service:
                logger.warning("[SIMILAR_SEARCH] FAISSIndexService not available")
                return []
            
            # Search FAISS index (provider='clip' required!)
            results = await faiss_service.search(
                provider='clip',
                query_vector=source_vector.tolist(),
                k=query.limit
            )
            logger.info(f"[SIMILAR_SEARCH] FAISS returned {len(results)} results")
            
            if not results:
                return []
            
            # Convert to FileRecords
            file_ids = [fid for fid, _dist in results]
            files = await FileRecord.find({"_id": {"$in": file_ids}})
            
            logger.info(f"[SIMILAR_SEARCH] Found {len(files)} similar images")
            return files
            
        except Exception as e:
            logger.error(f"[SIMILAR_SEARCH] Failed: {e}")
            import traceback
            logger.error(f"[SIMILAR_SEARCH] Traceback: {traceback.format_exc()}")
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

