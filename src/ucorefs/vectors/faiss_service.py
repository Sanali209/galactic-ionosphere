"""
UCoreFS - FAISS Index Service

In-memory vector index for fast similarity search.
Uses FAISS with MongoDB as persistent storage.
"""
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem


class FAISSIndexService(BaseSystem):
    """
    - Batch queries for efficiency
    - Lazy index building (on first search)
    """
    
    async def initialize(self) -> None:
        """Initialize FAISS service."""
        logger.info("FAISSIndexService initializing")
        
        self._faiss_available = False
        self._indexes: Dict[str, Any] = {}  # provider -> faiss.Index
        self._id_maps: Dict[str, Dict[int, ObjectId]] = {}  # provider -> {faiss_idx: file_id}
        self._reverse_maps: Dict[str, Dict[str, int]] = {}  # provider -> {file_id_str: faiss_idx}
        self._index_sizes: Dict[str, int] = {}
        
        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
            logger.info("FAISS library available")
        except ImportError:
            logger.warning("FAISS not available - using fallback linear search")
        
        await super().initialize()
        logger.info("FAISSIndexService ready")
    
    async def shutdown(self) -> None:
        """Shutdown FAISS service."""
        logger.info("FAISSIndexService shutting down")
        self._indexes.clear()
        self._id_maps.clear()
        await super().shutdown()
    
    def is_available(self) -> bool:
        """Check if FAISS is available."""
        return self._faiss_available
    
    async def build_index(self, provider: str, force: bool = False) -> bool:
        """
        Build or rebuild FAISS index for a provider.
        
        Args:
            provider: Embedding provider name (clip, blip, etc.)
            force: Force rebuild even if index exists
            
        Returns:
            True if successful
        """
        if not self._faiss_available:
            return False
        
        if provider in self._indexes and not force:
            return True
        
        try:
            from src.ucorefs.models.file_record import FileRecord
            import numpy as np
            
            logger.info(f"[FAISS] Building index for provider: {provider}")
            
            # Load all files with this provider's embeddings (UNIFIED STORAGE)
            files = await FileRecord.find({
                f"embeddings.{provider}": {"$exists": True},
                f"embeddings.{provider}.vector": {"$exists": True}
            })
            
            logger.info(f"[FAISS] Found {len(files)} files with {provider} embeddings")
            
            if not files:
                logger.warning(f"[FAISS] No embeddings found for provider: {provider}")
                return False
            
            # Extract vectors and build maps
            vectors = []
            id_map = {}
            reverse_map = {}
            
            for i, file in enumerate(files):
                vector = file.embeddings.get(provider, {}).get("vector")
                if vector and len(vector) > 0:
                    vectors.append(vector)
                    id_map[i] = file._id
                    reverse_map[str(file._id)] = i
            
            if not vectors:
                logger.error(f"[FAISS] All files had empty/invalid vectors for {provider}")
                return False
            
            # Convert to numpy
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # Normalize for cosine similarity (inner product on normalized vectors)
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vectors_np = vectors_np / norms
            
            # Build FAISS index
            dimension = vectors_np.shape[1]
            index = self._faiss.IndexFlatIP(dimension)  # Inner product
            index.add(vectors_np)
            
            # Store
            self._indexes[provider] = index
            self._id_maps[provider] = id_map
            self._reverse_maps[provider] = reverse_map
            self._index_sizes[provider] = len(vectors)
            
            logger.info(f"[FAISS] ✓ Built index for {provider}: {len(vectors)} vectors, dim={dimension}")
            return True
            
        except Exception as e:
            logger.error(f"[FAISS] Failed to build index for {provider}: {e}", exc_info=True)
            return False
    
    async def search(
        self,
        provider: str,
        query_vector: List[float],
        k: int = 10,
        file_ids: Optional[List[ObjectId]] = None
    ) -> List[Tuple[ObjectId, float]]:
        """
        Search for similar vectors.
        
        Args:
            provider: Embedding provider
            query_vector: Query embedding
            k: Number of results
            file_ids: Optional filter to search only these files
            
        Returns:
            List of (file_id, similarity_score) tuples
        """
        # Ensure index is built
        if provider not in self._indexes:
            logger.info(f"[FAISS] Index not found for {provider}, attempting to build...")
            await self.build_index(provider)
        
        if provider not in self._indexes:
            logger.warning(f"[FAISS] No index available for {provider} after build attempt")
            return []
        
        try:
            import numpy as np
            
            index = self._indexes[provider]
            id_map = self._id_maps[provider]
            
            logger.info(f"[FAISS] Searching index: {index.ntotal} vectors, k={k}")
            
            # Prepare query
            query = np.array([query_vector], dtype=np.float32)
            query = query / np.linalg.norm(query)  # Normalize
            
            # Search
            if file_ids:
                # Filter search - get more results and filter
                search_k = min(k * 10, index.ntotal)
                distances, indices = index.search(query, search_k)
                
                # Filter to requested file_ids
                file_id_set = {str(fid) for fid in file_ids}
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx >= 0:
                        fid = id_map.get(idx)
                        if fid and str(fid) in file_id_set:
                            results.append((fid, float(dist)))
                            if len(results) >= k:
                                break
            else:
                # Full search
                distances, indices = index.search(query, k)
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx >= 0:
                        fid = id_map.get(idx)
                        if fid:
                            results.append((fid, float(dist)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def add_vector(
        self,
        provider: str,
        file_id: ObjectId,
        vector: List[float]
    ) -> bool:
        """
        Add/update vector in index and MongoDB.
        
        Args:
            provider: Embedding provider
            file_id: File ObjectId
            vector: Embedding vector
            
        Returns:
            True if successful
        """
        try:
            from src.ucorefs.vectors.models import EmbeddingRecord
            from datetime import datetime
            
            # Upsert to MongoDB
            existing = await EmbeddingRecord.find_one({
                "file_id": file_id,
                "provider": provider
            })
            
            if existing:
                existing.vector = vector
                existing.dimension = len(vector)
                existing.updated_at = datetime.now()
                await existing.save()
            else:
                record = EmbeddingRecord(
                    file_id=file_id,
                    provider=provider,
                    vector=vector,
                    dimension=len(vector)
                )
                await record.save()
            
            # Mark index for rebuild (lazy - on next search)
            if provider in self._indexes:
                # For now, just mark dirty. Full rebuild is expensive.
                # In production, use incremental updates
                del self._indexes[provider]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            return False
    
    async def delete_vector(self, provider: str, file_id: ObjectId) -> bool:
        """Delete vector from index and MongoDB."""
        try:
            from src.ucorefs.vectors.models import EmbeddingRecord
            
            await EmbeddingRecord.find(
                {"file_id": file_id, "provider": provider}
            ).delete()
            
            # Mark index for rebuild
            if provider in self._indexes:
                del self._indexes[provider]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded indexes."""
        return {
            "faiss_available": self._faiss_available,
            "indexes": {
                provider: {
                    "size": self._index_sizes.get(provider, 0),
                    "loaded": provider in self._indexes
                }
                for provider in self._index_sizes
            }
        }
