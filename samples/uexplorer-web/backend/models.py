"""
Database Models for UExplorer Web

Beanie ORM models for MongoDB integration.
Based on the desktop UExplorer's UCoreFS data models.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from beanie import Document, Indexed
from pydantic import BaseModel, Field
from bson import ObjectId


# Enums
class ProcessingState(str, Enum):
    """File processing state"""
    RAW = "raw"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class RelationType(str, Enum):
    """File relationship types"""
    SIMILAR = "similar"
    DUPLICATE = "duplicate"
    RELATED = "related"
    CUSTOM = "custom"


# Base Models
class FSRecord(Document):
    """Base filesystem record"""
    path: Indexed(str, unique=True)
    name: str
    is_directory: bool
    parent_id: Optional[ObjectId] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    processing_state: ProcessingState = ProcessingState.RAW
    
    class Settings:
        name = "fs_records"
        indexes = ["path", "parent_id", "processing_state"]


class FileRecord(Document):
    """File metadata record"""
    path: Indexed(str, unique=True)
    name: str
    extension: Optional[str] = None
    size: int = 0
    mime_type: Optional[str] = None
    parent_id: Optional[ObjectId] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    file_modified_at: Optional[datetime] = None
    
    # Processing
    processing_state: ProcessingState = ProcessingState.RAW
    processing_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    rating: int = 0  # 0-5 stars
    custom_properties: Dict[str, Any] = Field(default_factory=dict)
    
    # AI Features
    description: Optional[str] = None
    tags_auto: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    class Settings:
        name = "file_records"
        indexes = ["path", "extension", "parent_id", "processing_state", "rating"]


class DirectoryRecord(Document):
    """Directory metadata record"""
    path: Indexed(str, unique=True)
    name: str
    parent_id: Optional[ObjectId] = None
    child_count: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    last_scanned_at: Optional[datetime] = None
    
    # State
    scan_state: str = "pending"  # pending, scanning, complete, error
    
    class Settings:
        name = "directory_records"
        indexes = ["path", "parent_id"]


# Tag System (MPPT - Modified Preorder Tree Traversal)
class Tag(Document):
    """Hierarchical tag with MPPT structure"""
    name: Indexed(str, unique=True)
    description: Optional[str] = None
    color: Optional[str] = None
    
    # MPPT structure
    parent_id: Optional[ObjectId] = None
    lft: int = 0
    rgt: int = 0
    level: int = 0
    
    # Statistics
    file_count: int = 0
    
    # Metadata
    synonyms: List[str] = Field(default_factory=list)
    antonyms: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "tags"
        indexes = ["name", "parent_id", "lft", "rgt"]


class FileTag(Document):
    """Many-to-many relationship between files and tags"""
    file_id: Indexed(ObjectId)
    tag_id: Indexed(ObjectId)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "file_tags"
        indexes = [
            [("file_id", 1), ("tag_id", 1)],  # Compound unique index
        ]


# Album System
class Album(Document):
    """Album (collection) of files"""
    name: Indexed(str, unique=True)
    description: Optional[str] = None
    icon: Optional[str] = None
    
    # Album type
    is_smart: bool = False
    query: Optional[Dict[str, Any]] = None  # Smart album query
    
    # Statistics
    file_count: int = 0
    
    # Metadata
    custom_properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "albums"
        indexes = ["name", "is_smart"]


class FileAlbum(Document):
    """Many-to-many relationship between files and albums"""
    file_id: Indexed(ObjectId)
    album_id: Indexed(ObjectId)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "file_albums"
        indexes = [
            [("file_id", 1), ("album_id", 1)],
        ]


# Detection System
class DetectionClass(Document):
    """Detection class (hierarchical)"""
    name: Indexed(str, unique=True)
    parent_id: Optional[ObjectId] = None
    
    # MPPT structure
    lft: int = 0
    rgt: int = 0
    level: int = 0
    
    # Metadata
    description: Optional[str] = None
    color: Optional[str] = None
    
    class Settings:
        name = "detection_classes"
        indexes = ["name", "parent_id", "lft", "rgt"]


class DetectionInstance(Document):
    """Detection bounding box for a file"""
    file_id: Indexed(ObjectId)
    class_id: Indexed(ObjectId)
    
    # Bounding box (normalized 0-1)
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    
    # Detection metadata
    confidence: float = 0.0
    backend: str = "unknown"  # yolo, mtcnn, etc.
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "detection_instances"
        indexes = ["file_id", "class_id"]


# Relation System
class Relation(Document):
    """Relationship between two files"""
    source_id: Indexed(ObjectId)
    target_id: Indexed(ObjectId)
    relation_type: RelationType
    
    # Metadata
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Feedback
    marked_wrong: bool = False
    
    class Settings:
        name = "relations"
        indexes = [
            "source_id", "target_id", "relation_type",
            [("source_id", 1), ("target_id", 1)],
        ]


# Embedding System
class EmbeddingRecord(Document):
    """Vector embedding for a file"""
    file_id: Indexed(ObjectId, unique=True)
    embedding: List[float]
    model: str = "clip"
    dimension: int = 768
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "embeddings"
        indexes = ["file_id", "model"]


# Annotation System
class AnnotationJob(Document):
    """Annotation job for files"""
    title: str
    description: Optional[str] = None
    
    # Files to annotate
    file_ids: List[ObjectId] = Field(default_factory=list)
    
    # Progress
    total_files: int = 0
    annotated_count: int = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "annotation_jobs"


class AnnotationRecord(Document):
    """Individual file annotation"""
    job_id: Indexed(ObjectId)
    file_id: Indexed(ObjectId)
    
    # Annotation data
    annotation_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    annotator: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "annotation_records"
        indexes = [
            "job_id", "file_id",
            [("job_id", 1), ("file_id", 1)],
        ]


# Rules Engine
class Rule(Document):
    """Automation rule"""
    name: str
    description: Optional[str] = None
    enabled: bool = True
    
    # Trigger
    trigger: str = "manual"  # on_import, on_tag, manual
    
    # Conditions & Actions (JSON)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Statistics
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "rules"


# Task System
class TaskRecord(Document):
    """Background task record"""
    task_id: Indexed(str, unique=True)
    task_type: str
    status: str = "pending"  # pending, running, completed, failed
    
    # Task data
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress
    progress: float = 0.0
    progress_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Settings:
        name = "task_records"
        indexes = ["task_id", "status", "task_type"]


# Audit Log
class JournalEvent(Document):
    """Audit log event"""
    event_type: str
    event_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    user: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "journal_events"
        indexes = ["event_type", "timestamp"]
