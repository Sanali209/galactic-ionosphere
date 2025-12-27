"""
UCoreFS - Relation Models

Models for file-to-file relations.
"""
from typing import Optional, Dict
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class RelationType(CollectionRecord):
    """
    Extensible relation type definition.
    
    Defines types like "image-image", "image-detection" with subtypes.
    Auto collection name: "relation_types"
    """
    # Type info
    type_name: str = Field(default="", index=True, unique=True)
    description: str = Field(default="")
    
    # Extensible subtypes list
    subtypes: list = Field(default_factory=list)
    
    def __str__(self) -> str:
        return f"RelationType: {self.type_name}"


class Relation(CollectionRecord):
    """
    File-to-file relation with extensible subtypes.
    
    Supports various relation types:
    - duplicate, near_duplicate, variant
    - wrong (user-marked incorrect relation)
    - custom subtypes
    
    Auto collection name: "relations"
    """
    # Files
    source_id: ObjectId = Field(default=None, index=True)
    target_id: ObjectId = Field(default=None, index=True)
    
    # Type
    relation_type_id: Optional[ObjectId] = Field(default=None, index=True)
    relation_type: str = Field(default="", index=True)  # Denormalized for speed
    subtype: str = Field(default="", index=True)
    
    # Validity
    is_valid: bool = Field(default=True, index=True)
    
    # Payload (threshold, score, system, etc.)
    payload: Dict = Field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Relation: {self.source_id} -> {self.target_id} ({self.relation_type}/{self.subtype})"
