"""
UCoreFS - Annotation Models

Models for annotation workflow supporting ML training data curation.

Supports:
- Binary classification (yes/no)
- Multi-class classification (one of N)
- Multi-label classification (many of N)
"""
from typing import Optional, List, Any
from datetime import datetime
from bson import ObjectId
from pydantic import Field as PydanticField

from src.core.database.orm import Field
from src.ucorefs.models.base import FSRecord


class AnnotationJob(FSRecord):
    """
    Annotation job definition.
    
    Defines an annotation task with available choices/labels.
    
    Auto collection name: "annotation_jobs"
    
    Example:
        # Binary classification
        job = AnnotationJob(
            name="NSFW Detection",
            job_type="binary",
            choices=["safe", "nsfw"]
        )
        
        # Multi-class
        job = AnnotationJob(
            name="Content Rating",
            job_type="multiclass", 
            choices=["general", "sensitive", "questionable", "explicit"]
        )
        
        # Multi-label
        job = AnnotationJob(
            name="Image Tags",
            job_type="multilabel",
            choices=["landscape", "portrait", "action", "still_life"]
        )
    """
    
    # Job type
    job_type: str = Field(default="binary", index=True)  # binary, multiclass, multilabel
    
    # Available choices/labels
    choices: List[str] = Field(default_factory=list)
    
    # Description
    description: str = Field(default="")
    
    # Statistics (updated periodically)
    total_files: int = Field(default=0)
    annotated_count: int = Field(default=0)
    
    # Created by
    created_by: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def __str__(self) -> str:
        return f"AnnotationJob: {self.name} ({self.job_type})"
    
    @property
    def progress_percent(self) -> float:
        """Get annotation progress as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.annotated_count / self.total_files) * 100
    
    @property
    def remaining_count(self) -> int:
        """Get count of unannotated files."""
        return max(0, self.total_files - self.annotated_count)


class AnnotationRecord(FSRecord):
    """
    Individual file annotation within a job.
    
    Links a file to an annotation job with the annotation value.
    
    Auto collection name: "annotation_records"
    """
    
    # Parent references
    job_id: ObjectId = Field(default=None, index=True)
    file_id: ObjectId = Field(default=None, index=True)
    
    # Annotation value
    # - Binary: "yes"/"no" or choice[0]/choice[1]
    # - Multiclass: single string from choices
    # - Multilabel: list of strings from choices
    value: Optional[Any] = Field(default=None)
    
    # Status
    is_annotated: bool = Field(default=False, index=True)
    skipped: bool = Field(default=False)
    
    # Metadata
    annotated_at: Optional[datetime] = Field(default=None)
    annotated_by: str = Field(default="")
    
    # Confidence (optional, for review)
    confidence: float = Field(default=1.0)  # 0.0-1.0
    
    # Notes
    notes: str = Field(default="")
    
    def __str__(self) -> str:
        status = "annotated" if self.is_annotated else "pending"
        return f"AnnotationRecord: {self.file_id} ({status})"
    
    def set_value(self, value: Any, annotated_by: str = "user") -> None:
        """Set annotation value and mark as annotated."""
        self.value = value
        self.is_annotated = True
        self.annotated_at = datetime.utcnow()
        self.annotated_by = annotated_by
    
    def clear_value(self) -> None:
        """Clear annotation and mark as pending."""
        self.value = None
        self.is_annotated = False
        self.annotated_at = None
