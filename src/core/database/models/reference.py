from typing import List, Optional, Dict
from bson import ObjectId
from src.core.database.orm import CollectionRecord, FieldPropInfo

class Reference(CollectionRecord, table="references", indexes=["source_id", "target_id", "type"]):
    """
    Universal graph edge representing a relationship between two entities.
    """
    source_id = FieldPropInfo("source_id", field_type=ObjectId)
    target_id = FieldPropInfo("target_id", field_type=ObjectId)
    rel_type = FieldPropInfo("type", default="related", field_type=str)
    
    # Metadata for the relation (e.g. similarity score, polygon points)
    payload = FieldPropInfo("payload", default={}, field_type=dict)

class RelationManager:
    """
    Service for managing relationships.
    """
    
    @staticmethod
    async def link(source: ObjectId, target: ObjectId, rel_type: str, payload: Dict = None) -> Reference:
        if not payload:
            payload = {}
            
        # Explicitly assign via property to handle FieldPropInfo name mapping correctly
        ref = Reference(source_id=source, target_id=target)
        ref.rel_type = rel_type
        ref.payload = payload
        
        await ref.save()
        return ref

    @staticmethod
    async def get_related(source: ObjectId, rel_type: str = None) -> List[Reference]:
        """Finds all outgoing edges from source."""
        query = {"source_id": source}
        if rel_type:
            query["type"] = rel_type
        return await Reference.find(query)

    @staticmethod
    async def get_incoming(target: ObjectId, rel_type: str = None) -> List[Reference]:
        """Finds all incoming edges to target (back-references)."""
        query = {"target_id": target}
        if rel_type:
            query["type"] = rel_type
        return await Reference.find(query)

    @staticmethod
    async def unlink(source: ObjectId, target: ObjectId, rel_type: str):
        """Removes a specific link."""
        # Find and delete
        refs = await Reference.find({"source_id": source, "target_id": target, "type": rel_type})
        for r in refs:
            await r.delete()
