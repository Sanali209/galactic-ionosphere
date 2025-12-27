"""
UCoreFS - Annotation Package

Provides annotation workflow for ML training data curation.
"""
from src.ucorefs.annotation.models import AnnotationJob, AnnotationRecord
from src.ucorefs.annotation.service import AnnotationService

__all__ = [
    "AnnotationJob",
    "AnnotationRecord",
    "AnnotationService",
]

