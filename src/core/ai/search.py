from typing import List, Dict, Any
from loguru import logger
from bson import ObjectId

from src.core.database.models.image import ImageRecord
from src.core.ai.vector_driver import VectorDriver
from src.core.ai.service import EmbeddingService

class SearchService:
    """
    High-level search interface.
    """
    def __init__(self, vector_driver: VectorDriver, embedding_service: EmbeddingService):
        self.vector_driver = vector_driver
        self.embedding_service = embedding_service

    async def search_by_text(self, query: str, limit: int = 20) -> List[ImageRecord]:
        """
        Semantic search using text query.
        """
        # 1. Encode text
        vector = await self.embedding_service.encode_text(query)
        # Ensure we have a valid non-empty vector
        if vector is None or getattr(vector, 'size', 0) == 0:
            logger.warning("Empty vector generated for query.")
            return []

        # 2. Search Vector DB
        # returns list of ScoredPoint (id, score, payload, etc.)
        results = await self.vector_driver.search(vector, limit=limit)
        
        if not results:
            return []
            
        # 3. Extract IDs and Hydrate
        # Qdrant IDs are stored as UUID string (converted from ObjectId).
        # We need to reverse mapping? 
        # Actually, if we stored `uuid5(NAMESPACE_OID, str(oid))`, we can't easily reverse it without a lookup map.
        # UNLESS we stored the original ObjectId in the payload!
        # Let's check `AITaskHandlers`: 
        # payload = {"rating": ..., "tags": ...} -> We didn't store original ID explicitly in payload usually needed? 
        # Wait, if we used a deterministic UUID, we can't reverse.
        # FIX: We MUST store the original 'mongo_id' in the Qdrant payload to retrieve it.
        
        # We need to update AITaskHandlers to store 'mongo_id' in payload.
        # But for now, let's assume we fix that or extract it if we used the ID as string directly?
        # Qdrant supports string IDs (which look like UUIDs).
        
        # Let's assume we add `mongo_id` to payload in handler. I will fix Handler next/now.
        # For now, let's write code assuming payload["mongo_id"] exists.
        
        mongo_ids = []
        for point in results:
            # point might be ScoredPoint or dict depending on client version/mock
            # Assuming object access
            if hasattr(point, 'payload') and point.payload and 'mongo_id' in point.payload:
                mongo_ids.append(ObjectId(point.payload['mongo_id']))
        
        if not mongo_ids:
            return []
            
        # 4. Fetch from DB
        # Use $in query, but we want to preserve order of relevance!
        # .find() does not guarantee order match to $in list.
        # We need to map results back to order.
        
        records = await ImageRecord.find({"_id": {"$in": mongo_ids}})
        record_map = {rec.id: rec for rec in records}
        
        ordered_records = []
        for mid in mongo_ids:
            if mid in record_map:
                ordered_records.append(record_map[mid])
                
        return ordered_records
