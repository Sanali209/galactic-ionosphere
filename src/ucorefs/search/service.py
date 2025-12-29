"""
UCoreFS - Search Service

Unified search combining MongoDB text/filter queries with FAISS vector similarity.
Single entry point for all file search operations.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem


@dataclass
class SearchResult:
    """Individual search result with scoring."""
    file_id: ObjectId
    score: float = 1.0
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    match_type: str = "filter"  # filter, text, vector, hybrid


@dataclass
class SearchQuery:
    """
    Unified search query parameters.
    
    Supports:
    - Text search (file name, description, tags)
    - MongoDB filters (file_type, rating, label, etc.)
    - Vector similarity (CLIP or BLIP embeddings)
    """
    # Text query
    text: Optional[str] = None
    
    # MongoDB filters
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Vector search
    vector_search: bool = False
    vector_provider: str = "clip"  # clip, blip, mobilenet
    vector_query: Optional[List[float]] = None  # Pre-computed embedding
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    # Sorting
    sort_by: str = "score"  # score, name, modified_at, rating
    sort_desc: bool = True


class SearchService(BaseSystem):
    """
            vector_provider="clip"
        )
        results = await search_service.search(query)
    """
    
    async def initialize(self) -> None:
        """Initialize search service."""
        logger.info("SearchService initializing")
        
        # Get dependencies
        self._faiss_service = None
        self._vector_service = None
        
        try:
            from src.ucorefs.vectors.faiss_service import FAISSIndexService
            self._faiss_service = self.locator.get_system(FAISSIndexService)
        except (KeyError, ImportError):
            logger.info("FAISSIndexService not available - vector search disabled")
        
        try:
            from src.ucorefs.vectors.service import VectorService
            self._vector_service = self.locator.get_system(VectorService)
        except (KeyError, ImportError):
            pass
        
        await super().initialize()
        logger.info("SearchService ready")
    
    async def shutdown(self) -> None:
        """Shutdown search service."""
        logger.info("SearchService shutting down")
        await super().shutdown()
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute unified search.
        
        Args:
            query: SearchQuery with all parameters
            
        Returns:
            List of SearchResult sorted by score
        """
        from src.ucorefs.models.file_record import FileRecord
        
        results_map: Dict[str, SearchResult] = {}
        
        # Step 1: MongoDB filter search
        mongo_filter = self._build_mongo_filter(query)
        
        if query.text:
            # Text search on name, description, tags
            mongo_filter["$or"] = [
                {"name": {"$regex": query.text, "$options": "i"}},
                {"description": {"$regex": query.text, "$options": "i"}},
                {"ai_description": {"$regex": query.text, "$options": "i"}},
            ]
        
        # Execute MongoDB query
        files = await FileRecord.find(
            mongo_filter,
            limit=query.limit * 2 if query.vector_search else query.limit
        ).to_list()
        
        # Add to results
        for file in files:
            file_id_str = str(file._id)
            text_score = 1.0
            
            # Boost exact name matches
            if query.text and query.text.lower() in file.name.lower():
                text_score = 1.5
            
            results_map[file_id_str] = SearchResult(
                file_id=file._id,
                score=text_score,
                text_score=text_score,
                match_type="text" if query.text else "filter"
            )
        
        # Step 2: Vector similarity search
        if query.vector_search and self._faiss_service:
            vector_results = await self._vector_search(query, list(results_map.keys()))
            
            # Merge vector scores
            for file_id, vector_score in vector_results:
                file_id_str = str(file_id)
                
                if file_id_str in results_map:
                    # Hybrid: combine text and vector scores
                    existing = results_map[file_id_str]
                    existing.vector_score = vector_score
                    existing.score = (existing.text_score or 1.0) * 0.4 + vector_score * 0.6
                    existing.match_type = "hybrid"
                else:
                    # Vector-only result
                    results_map[file_id_str] = SearchResult(
                        file_id=file_id,
                        score=vector_score,
                        vector_score=vector_score,
                        match_type="vector"
                    )
        
        # Step 3: Sort and paginate
        results = list(results_map.values())
        
        if query.sort_by == "score":
            results.sort(key=lambda r: r.score, reverse=query.sort_desc)
        
        # Apply offset/limit
        return results[query.offset:query.offset + query.limit]
    
    async def search_similar(
        self,
        file_id: ObjectId,
        provider: str = "clip",
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find files similar to a given file.
        
        Args:
            file_id: Source file ObjectId
            provider: Embedding provider
            limit: Max results
            
        Returns:
            List of similar files
        """
        if not self._faiss_service:
            return []
        
        try:
            from src.ucorefs.vectors.models import EmbeddingRecord
            
            # Get source embedding
            embedding = await EmbeddingRecord.find_one({
                "file_id": file_id,
                "provider": provider
            })
            
            if not embedding:
                logger.debug(f"No {provider} embedding for {file_id}")
                return []
            
            # Search similar
            similar = await self._faiss_service.search(
                provider,
                embedding.vector,
                k=limit + 1  # +1 to exclude self
            )
            
            # Filter out source file and format results
            results = []
            for fid, score in similar:
                if fid != file_id:
                    results.append(SearchResult(
                        file_id=fid,
                        score=score,
                        vector_score=score,
                        match_type="vector"
                    ))
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return []
    
    async def _vector_search(
        self,
        query: SearchQuery,
        filtered_ids: List[str]
    ) -> List[tuple]:
        """Execute vector similarity search."""
        if not self._faiss_service:
            return []
        
        # Get query embedding
        query_vector = query.vector_query
        
        if not query_vector and query.text:
            # Generate embedding from text (requires embedding service)
            query_vector = await self._get_text_embedding(query.text, query.vector_provider)
        
        if not query_vector:
            return []
        
        # Search with optional filtering
        file_ids = [ObjectId(fid) for fid in filtered_ids] if filtered_ids else None
        
        return await self._faiss_service.search(
            query.vector_provider,
            query_vector,
            k=query.limit,
            file_ids=file_ids
        )
    
    async def _get_text_embedding(self, text: str, provider: str) -> Optional[List[float]]:
        """Get embedding for text query using CLIPExtractor."""
        if provider == "clip":
            try:
                from src.ucorefs.extractors.clip_extractor import CLIPExtractor
                
                extractor = CLIPExtractor(self.locator)
                embedding = await extractor.encode_text(text)
                if embedding:
                    return embedding
            except Exception as e:
                logger.error(f"CLIP text encoding failed: {e}")
        
        logger.debug(f"Text embedding not available for {provider}")
        return None
    
    def _build_mongo_filter(self, query: SearchQuery) -> Dict[str, Any]:
        """Build MongoDB filter from query parameters."""
        mongo_filter = {}
        
        for key, value in query.filters.items():
            if value is not None:
                if key == "tag_ids" and isinstance(value, list):
                    mongo_filter["tag_ids"] = {"$in": value}
                elif key == "rating_min":
                    mongo_filter["rating"] = {"$gte": value}
                elif key == "file_types" and isinstance(value, list):
                    mongo_filter["file_type"] = {"$in": value}
                else:
                    mongo_filter[key] = value
        
        return mongo_filter
