"""
UCoreFS - Filesystem Database Models

Base classes for filesystem records.
"""
from datetime import datetime
from typing import Optional, List
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field, ReferenceField


class FSRecord(CollectionRecord):
    """
    Base class for all filesystem records (files and directories).
    
    Provides common fields for path, hierarchy, and metadata tracking.
    Auto collection name: "fs_records"
    """
    # Path information
    path: str = Field(default="", index=True)
    name: str = Field(default="", index=True)
    
    # Hierarchy
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    root_id: Optional[ObjectId] = Field(default=None, index=True)
    
    # Type information
    is_virtual: bool = Field(default=False)
    driver_type: str = Field(default="default")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    discovered_at: datetime = Field(default_factory=datetime.now)
    
    # Size
    size_bytes: int = Field(default=0)
    
    def __str__(self) -> str:
        return f"FSRecord: {self.name} ({self.path})"
