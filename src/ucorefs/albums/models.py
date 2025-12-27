"""
UCoreFS - Album Models

Album system with smart albums (dynamic queries).
"""
from typing import Optional, Dict, Any
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class Album(CollectionRecord):
    """
    Album for organizing files.
    
    Supports:
    - Manual albums (user-curated)
    - Smart albums (dynamic query-based)
    - Hierarchical organization
    
    Auto collection name: "albums"
    """
    # Album info
    name: str = Field(default="", index=True)
    description: str = Field(default="")
    
    # Hierarchy
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    
    # Cover image
    cover_id: Optional[ObjectId] = Field(default=None)
    
    # Smart album
    is_smart: bool = Field(default=False, index=True)
    smart_query: Dict[str, Any] = Field(default_factory=dict)
    
    # Manual album file list (for non-smart albums)
    file_ids: list = Field(default_factory=list)
    
    # Statistics
    file_count: int = Field(default=0)
    
    def __str__(self) -> str:
        smart_str = " [Smart]" if self.is_smart else ""
        return f"Album: {self.name}{smart_str} ({self.file_count} files)"
