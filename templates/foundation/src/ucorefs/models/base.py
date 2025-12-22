"""
UCoreFS - Filesystem Database Models

Base classes for filesystem records.
"""
from datetime import datetime
from enum import IntEnum
from typing import Optional, List, Dict, Any
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field, ReferenceField


class ProcessingState(IntEnum):
    """
    File processing stage for progressive loading.
    
    Files progress through states as background processing completes:
    - Phase 1: DISCOVERED → REGISTERED (instant, batch 200)
    - Phase 2: METADATA_READY → THUMBNAIL_READY → INDEXED (batch 20)
    - Phase 3: ANALYZED → COMPLETE (batch 1)
    """
    DISCOVERED = 0       # Just found on filesystem
    REGISTERED = 10      # Path, name, size stored
    METADATA_READY = 20  # EXIF, XMP extracted
    THUMBNAIL_READY = 30 # Thumbnails generated
    INDEXED = 40         # Basic embeddings stored
    ANALYZED = 50        # Detections, AI analysis done
    COMPLETE = 100       # All processing finished


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
    
    # Processing state (for progressive loading)
    processing_state: int = Field(default=ProcessingState.DISCOVERED, index=True)
    processing_errors: List[str] = Field(default_factory=list)
    last_processed_at: Optional[datetime] = Field(default=None)
    
    def __str__(self) -> str:
        return f"FSRecord: {self.name} ({self.path})"

