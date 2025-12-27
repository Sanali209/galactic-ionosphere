"""
UCoreFS - Tag Models

Hierarchical tag system with synonyms and antonyms.
"""
from typing import List, Optional
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class Tag(CollectionRecord):
    """
    Hierarchical tag with MPTT structure.
    
    Supports:
    - Hierarchical organization (Animals/Mammals/Cats)
    - Synonyms (cat <-> feline <-> kitty)
    - Antonyms (work <-> personal)
    
    Auto collection name: "tags"
    """
    # Tag info
    name: str = Field(default="", index=True)
    full_path: str = Field(default="", index=True)
    
    # Hierarchy (MPTT)
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    lft: int = Field(default=0, index=True)
    rgt: int = Field(default=0, index=True)
    depth: int = Field(default=0)
    
    # Statistics
    file_count: int = Field(default=0)
    
    # Synonyms and antonyms
    synonym_ids: List[ObjectId] = Field(default_factory=list)
    antonym_ids: List[ObjectId] = Field(default_factory=list)
    
    # Color for UI
    color: str = Field(default="")
    
    def __str__(self) -> str:
        return f"Tag: {self.full_path} ({self.file_count} files)"
