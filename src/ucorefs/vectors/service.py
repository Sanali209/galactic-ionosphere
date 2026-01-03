"""
UCoreFS - Vector Service

Vector storage and similarity search using FAISS + MongoDB.
"""
from typing import List, Dict, Any, Optional
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.service_decorator import Service


@Service
class VectorService(BaseSystem):
    """
    Vector storage service using FAISS + MongoDB.
    
    .. deprecated::
        This service is being deprecated. Use `SearchService` for search
        and `FAISSIndexService` directly for vector operations.
    
    Architecture:
    - Embeddings stored in MongoDB (EmbeddingRecord)
    - FAISS provides in-memory similarity search
    - Supports multiple providers (clip, blip, mobilenet)
    
    Features:
    - Multi-provider embeddings per file
    - Cosine similarity search
    - Metadata filtering via MongoDB
    """
    
    depends_on = ["FAISSIndexService", "DatabaseManager"]
    
    PROVIDERS = ["clip", "blip", "mobilenet", "thumb"]
    
    async def initialize(self) -> None:
        """Initialize vector service with FAISS backend."""
        # Get or create FAISSIndexService
        self._faiss_service = None
        try:
            from src.ucorefs.vectors.faiss_service import FAISSIndexService
            self._faiss_service = self.locator.get_system(FAISSIndexService)
        except (KeyError, ImportError):
            logger.info("FAISSIndexService not registered - will use fallback")
        
        await super().initialize()
    
    async def shutdown(self) -> None:
        """Shutdown vector service."""
        await super().shutdown()
    
    def is_available(self) -> bool:
        """Check if vector search is available."""
        return self._faiss_service is not None and self._faiss_service.is_available()
    
    async def upsert(
        self,
        collection: str,
        file_id: ObjectId,
        vector: List[float],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Insert or update vector embedding.
        
        Args:
            collection: Provider name (clip, blip, etc.)
            file_id: File ObjectId
            vector: Embedding vector
            metadata: Optional metadata (stored in FileRecord, not here)
            
        Returns:
            True if successful
        """
        if not self._faiss_service:
            return False
        
        return await self._faiss_service.add_vector(collection, file_id, vector)
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection: Provider name
            query_vector: Query embedding
            filters: MongoDB filters to pre-filter files
            limit: Max results
            
        Returns:
            List of results with file_id, score, and metadata
        """
        if not self._faiss_service:
            return []
        
        # If filters provided, first get filtered file_ids from MongoDB
        file_ids = None
        if filters:
            file_ids = await self._get_filtered_file_ids(filters, limit * 10)
        
        # FAISS search
        results = await self._faiss_service.search(
            collection, 
            query_vector, 
            k=limit,
            file_ids=file_ids
        )
        
        # Format results
        formatted = []
        for file_id, score in results:
            formatted.append({
                "file_id": file_id,
                "score": score,
                "distance": 1.0 - score  # Convert similarity to distance
            })
        
        return formatted
    
    async def delete(self, collection: str, file_id: ObjectId) -> bool:
        """Delete vector for a file."""
        if not self._faiss_service:
            return False
        
        return await self._faiss_service.delete_vector(collection, file_id)
    
    async def _get_filtered_file_ids(
        self, 
        filters: Dict[str, Any],
        limit: int
    ) -> List[ObjectId]:
        """Get file IDs matching filters from MongoDB."""
        try:
            from src.ucorefs.models.file_record import FileRecord
            
            files = await FileRecord.find(filters, limit=limit).to_list()
            return [f._id for f in files]
            
        except Exception as e:
            logger.error(f"Failed to get filtered file IDs: {e}")
            return []
    
    async def build_metadata_payload(self, file_id: ObjectId) -> Dict[str, Any]:
        """
        Build metadata payload from MongoDB record.
        
        Note: With FAISS backend, metadata is stored in FileRecord,
        not duplicated to vector store.
        """
        try:
            from src.ucorefs.models.file_record import FileRecord
            from pathlib import Path
            
            file = await FileRecord.get(file_id)
            if not file:
                return {}
            
            directory_path = str(Path(file.path).parent)
            
            return {
                "file_id": str(file_id),
                "file_type": file.file_type,
                "extension": file.extension,
                "label": file.label,
                "name": file.name,
                "directory_path": directory_path,
                "rating": file.rating,
                "tags": [str(tid) for tid in file.tag_ids],
                "created_at": file.created_at.isoformat() if file.created_at else ""
            }
        
        except Exception as e:
            logger.error(f"Failed to build metadata payload: {e}")
            return {}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector service statistics."""
        if self._faiss_service:
            return await self._faiss_service.get_index_stats()
        return {"faiss_available": False}

