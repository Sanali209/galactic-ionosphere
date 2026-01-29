"""
Relations API Routes

File relationship management (similar, duplicate, etc.)
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime

from models import Relation, RelationType, FileRecord
from loguru import logger


router = APIRouter(prefix="/api/relations", tags=["Relations"])


# Request/Response Models
class RelationCreate(BaseModel):
    source_id: str
    target_id: str
    relation_type: str  # similar, duplicate, related
    confidence: float = 0.0
    metadata: dict = {}


class RelationResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    marked_wrong: bool
    created_at: str


@router.get("/", response_model=List[RelationResponse])
async def list_relations(
    relation_type: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """List relations with optional filtering"""
    query = {}
    if relation_type:
        query["relation_type"] = relation_type
    
    relations = await Relation.find(query).limit(limit).to_list()
    
    return [
        RelationResponse(
            id=str(rel.id),
            source_id=str(rel.source_id),
            target_id=str(rel.target_id),
            relation_type=rel.relation_type.value,
            confidence=rel.confidence,
            marked_wrong=rel.marked_wrong,
            created_at=rel.created_at.isoformat()
        )
        for rel in relations
    ]


@router.post("/", response_model=RelationResponse)
async def create_relation(relation_data: RelationCreate):
    """Create a new relation between files"""
    # Validate file IDs
    source = await FileRecord.get(ObjectId(relation_data.source_id))
    target = await FileRecord.get(ObjectId(relation_data.target_id))
    
    if not source or not target:
        raise HTTPException(status_code=404, detail="One or both files not found")
    
    # Check if relation already exists
    existing = await Relation.find_one({
        "source_id": ObjectId(relation_data.source_id),
        "target_id": ObjectId(relation_data.target_id),
        "relation_type": relation_data.relation_type
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Relation already exists")
    
    # Create relation
    relation = Relation(
        source_id=ObjectId(relation_data.source_id),
        target_id=ObjectId(relation_data.target_id),
        relation_type=RelationType(relation_data.relation_type),
        confidence=relation_data.confidence,
        metadata=relation_data.metadata
    )
    await relation.insert()
    
    logger.info(f"Created {relation_data.relation_type} relation: {relation_data.source_id} -> {relation_data.target_id}")
    
    return RelationResponse(
        id=str(relation.id),
        source_id=str(relation.source_id),
        target_id=str(relation.target_id),
        relation_type=relation.relation_type.value,
        confidence=relation.confidence,
        marked_wrong=relation.marked_wrong,
        created_at=relation.created_at.isoformat()
    )


@router.get("/file/{file_id}")
async def get_file_relations(file_id: str):
    """Get all relations for a file"""
    file_obj_id = ObjectId(file_id)
    
    # Get relations where file is source or target
    relations = await Relation.find({
        "$or": [
            {"source_id": file_obj_id},
            {"target_id": file_obj_id}
        ]
    }).to_list()
    
    result = []
    for rel in relations:
        # Get related file info
        related_id = rel.target_id if rel.source_id == file_obj_id else rel.source_id
        related_file = await FileRecord.get(related_id)
        
        if related_file:
            result.append({
                "relation_id": str(rel.id),
                "relation_type": rel.relation_type.value,
                "confidence": rel.confidence,
                "related_file": {
                    "id": str(related_file.id),
                    "name": related_file.name,
                    "path": related_file.path,
                    "size": related_file.size
                },
                "is_source": rel.source_id == file_obj_id
            })
    
    return {
        "file_id": file_id,
        "relations": result,
        "count": len(result)
    }


@router.delete("/{relation_id}")
async def delete_relation(relation_id: str):
    """Delete a relation"""
    relation = await Relation.get(ObjectId(relation_id))
    if not relation:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    await relation.delete()
    logger.info(f"Deleted relation: {relation_id}")
    
    return {"success": True, "deleted_id": relation_id}


@router.put("/{relation_id}/mark-wrong")
async def mark_relation_wrong(relation_id: str):
    """Mark a relation as incorrect"""
    relation = await Relation.get(ObjectId(relation_id))
    if not relation:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    relation.marked_wrong = True
    await relation.save()
    
    return {"success": True, "relation_id": relation_id, "marked_wrong": True}
