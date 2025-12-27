"""
UCoreFS - Detection Models

Models for detection instances (face, object bounding boxes).
"""
from typing import Optional, Dict
from bson import ObjectId

from src.core.database.orm import Field
from src.ucorefs.models.base import FSRecord


class DetectionClass(FSRecord):
    """
    Hierarchical detection class (face, car, character).
    
    Uses MPTT-like structure for hierarchy.
    Auto collection name: "detection_classes"
    """
    # Hierarchy
    parent_class_id: Optional[ObjectId] = Field(default=None, index=True)
    lft: int = Field(default=0)  # MPTT left
    rgt: int = Field(default=0)  # MPTT right
    
    # Class info
    class_name: str = Field(default="", index=True)
    
    def __str__(self) -> str:
        return f"DetectionClass: {self.class_name}"


class DetectionObject(FSRecord):
    """
    Named detection object (person name, car manufacturer).
    
    Auto collection name: "detection_objects"
    """
    # Class reference
    detection_class_id: ObjectId = Field(default=None, index=True)
    
    # Object info
    object_name: str = Field(default="", index=True)
    
    def __str__(self) -> str:
        return f"DetectionObject: {object_name}"


class DetectionInstance(FSRecord):
    """
    Detection instance on a file (virtual FSRecord).
    
    Represents a detected region with bounding box.
    Searchable and filterable like regular files.
    
    Auto collection name: "detection_instances"
    """
    # Parent file
    file_id: ObjectId = Field(default=None, index=True)
    
    # Classification
    detection_class_id: Optional[ObjectId] = Field(default=None, index=True)
    detection_object_id: Optional[ObjectId] = Field(default=None, index=True)
    
    # Bounding box (normalized 0-1)
    bbox: Dict = Field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})
    
    # Confidence
    confidence: float = Field(default=0.0)
    
    # Embedding (stored in ChromaDB)
    embedding_id: str = Field(default="")
    
    def __init__(self, **kwargs):
        # Mark as virtual by default
        kwargs.setdefault('is_virtual', True)
        kwargs.setdefault('driver_type', 'detection')
        super().__init__(**kwargs)
    
    def __str__(self) -> str:
        return f"Detection: {self.name} at {self.bbox}"
