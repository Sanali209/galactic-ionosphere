from typing import List, Dict, Any, Optional
import asyncio
from loguru import logger
# Import localized to avoid crash if dependencies missing during early load
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    QdrantClient = None
    models = None

class VectorDriver:
    """
    Manages interactions with Qdrant Vector DB.
    """
    def __init__(self, host: str, port: int, collection_name: str = "gallery"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        
        # Dimensions for standard CLIP (ViT-B/32) is 512.
        self.vector_size = 512 

    @staticmethod
    def to_qdrant_id(mongo_id: str) -> str:
        """Converts a Mongo ObjectId string to a UUIDv5 (OID Namespace)."""
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_OID, str(mongo_id))) 

    def connect(self):
        """Synchronous connection initialization (QdrantClient is sync/async hybrid)."""
        if not QdrantClient:
             logger.error("qdrant-client not installed.")
             return

        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection(self):
        """Checks if collection exists, else creates it."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")

    async def upsert_vector(self, point_id: str, vector: List[float], payload: Dict[str, Any] = None):
        """
        Upserts a vector.
        point_id: Typically the specific Hex UUID (or we convert ObjectId to UUID).
        """
        if not self.client:
             logger.warning("Qdrant client not initialized.")
             return

        try:
            # Create a PointStruct
            point = models.PointStruct(
                id=point_id, 
                vector=vector,
                payload=payload or {}
            )
            
            # Using asyncio.to_thread for the sync client call
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[point]
            )
        except Exception as e:
            logger.error(f"Error upserting vector for {point_id}: {e}")

    async def search(self, vector: List[float], limit: int = 10):
        if not self.client:
            return []
            
        try:
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=vector,
                limit=limit
            )
            return results.points # query_points returns a QueryResponse with points
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
