"""
UCoreFS - Vector Service

ChromaDB integration for vector storage and hybrid search.
"""
from typing import List, Dict, Any, Optional
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem


class VectorService(BaseSystem):
    """
    Vector storage service using ChromaDB.
    
    Features:
    - CLIP image embeddings
    - BLIP text embeddings
    - Thumbnail vector embeddings
    - Hybrid search (vector + metadata filtering)
    - Payload duplication from MongoDB
    """
    
    COLLECTIONS = {
        "file_embeddings": "Primary file embeddings (CLIP)",
        "text_embeddings": "Text content embeddings (BLIP)",
        "thumb_embeddings": "Thumbnail embeddings"
    }
    
    async def initialize(self) -> None:
        """Initialize vector service and ChromaDB."""
        logger.info("VectorService initializing")
        
        self._chroma_available = False
        self._client = None
        self._collections = {}
        
        try:
            import chromadb
            
            # Initialize ChromaDB client with new persistent API
            self._client = chromadb.PersistentClient(path="./chromadb")
            
            # Create/get collections
            for coll_name, description in self.COLLECTIONS.items():
                self._collections[coll_name] = self._client.get_or_create_collection(
                    name=coll_name,
                    metadata={"description": description}
                )
            
            self._chroma_available = True
            logger.info("ChromaDB initialized successfully")
        
        except ImportError:
            logger.warning("ChromaDB not available, vector search disabled")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
        
        await super().initialize()
    
    async def shutdown(self) -> None:
        """Shutdown vector service."""
        logger.info("VectorService shutting down")
        
        if self._client:
            try:
                # ChromaDB client doesn't need explicit close
                pass
            except Exception as e:
                logger.error(f"Error during ChromaDB shutdown: {e}")
        
        await super().shutdown()
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        return self._chroma_available
    
    async def upsert(
        self,
        collection: str,
        file_id: ObjectId,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Insert or update vector with metadata.
        
        Args:
            collection: Collection name
            file_id: File ObjectId
            vector: Embedding vector
            metadata: Metadata payload (duplicated from MongoDB)
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        if collection not in self._collections:
            logger.error(f"Unknown collection: {collection}")
            return False
        
        try:
            coll = self._collections[collection]
            file_id_str = str(file_id)
            
            # Upsert (ChromaDB handles both insert and update)
            coll.upsert(
                ids=[file_id_str],
                embeddings=[vector],
                metadatas=[metadata]
            )
            
            logger.debug(f"Upserted vector for {file_id_str} in {collection}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            return False
    
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
            collection: Collection name
            query_vector: Query embedding vector
            filters: Metadata filters (where clause)
            limit: Max results
            
        Returns:
            List of results with ids, distances, and metadata
        """
        if not self.is_available():
            return []
        
        if collection not in self._collections:
            logger.error(f"Unknown collection: {collection}")
            return []
        
        try:
            coll = self._collections[collection]
            
            # Query ChromaDB
            results = coll.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=filters if filters else None
            )
            
            # Format results
            formatted = []
            if results and results['ids']:
                for i, file_id_str in enumerate(results['ids'][0]):
                    formatted.append({
                        "file_id": ObjectId(file_id_str),
                        "distance": results['distances'][0][i],
                        "score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    async def delete(self, collection: str, file_id: ObjectId) -> bool:
        """
        Delete vector for a file.
        
        Args:
            collection: Collection name
            file_id: File ObjectId
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        if collection not in self._collections:
            return False
        
        try:
            coll = self._collections[collection]
            coll.delete(ids=[str(file_id)])
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            return False
    
    async def build_metadata_payload(self, file_id: ObjectId) -> Dict[str, Any]:
        """
        Build metadata payload from MongoDB record.
        
        Duplicates key fields for filtering in ChromaDB.
        
        Args:
            file_id: File ObjectId
            
        Returns:
            Metadata dict
        """
        try:
            from src.ucorefs.models.file_record import FileRecord
            
            file = await FileRecord.get(file_id)
            if not file:
                return {}
            
            # Extract directory path
            from pathlib import Path
            directory_path = str(Path(file.path).parent)
            
            return {
                "file_id": str(file_id),
                "file_type": file.file_type,
                "extension": file.extension,
                "label": file.label,
                "name": file.name,
                "directory_path": directory_path,
                "rating": file.rating,
                "tags": [str(tid) for tid in file.tag_ids],  # Tag IDs as strings
                "created_at": file.created_at.isoformat() if file.created_at else ""
            }
        
        except Exception as e:
            logger.error(f"Failed to build metadata payload: {e}")
            return {}
