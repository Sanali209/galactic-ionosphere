"""
UCoreFS - FileRecord Model

Represents a file in the filesystem database.
"""
from datetime import datetime
from typing import Optional, List
from bson import ObjectId

from src.core.database.orm import Field
from src.ucorefs.models.base import FSRecord


class FileRecord(FSRecord):
    """
    Represents a file in the filesystem.
    
    Extends FSRecord with file-specific metadata.
    Auto collection name: "file_records"
    """
    # File type information
    file_type: str = Field(default="unknown", index=True)
    mime_type: str = Field(default="")
    extension: str = Field(default="", index=True)
    hash_md5: str = Field(default="", index=True)
    
    # User metadata
    favorite: bool = Field(default=False, index=True)
    rating: int = Field(default=0, index=True)  # 0-5
    label: str = Field(default="", index=True)  # Color label
    description: str = Field(default="")
    
    # AI-generated metadata
    ai_description: str = Field(default="")
    ai_caption: str = Field(default="")  # BLIP caption
    
    # Processing state
    has_thumbnail: bool = Field(default=False)
    has_vector: bool = Field(default=False)
    
    # Tags and albums (stored as IDs for querying)
    tag_ids: List[ObjectId] = Field(default_factory=list)
    album_ids: List[ObjectId] = Field(default_factory=list)
    
    def __str__(self) -> str:
        return f"File: {self.name} ({self.file_type})"
