"""
UCoreFS - Rule Conditions

Extensible condition system for rules.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from loguru import logger

from src.ucorefs.models.file_record import FileRecord


class ICondition(ABC):
    """
    Interface for rule conditions.
    
    Conditions evaluate to True/False based on file record.
    """
    
    @abstractmethod
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        """
        Evaluate condition against file.
        
        Args:
            file: FileRecord to evaluate
            context: Additional context data
            
        Returns:
            True if condition matches
        """
        pass


class PathContainsCondition(ICondition):
    """Condition: path contains substring."""
    
    def __init__(self, substring: str):
        self.substring = substring
    
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        return self.substring.lower() in file.path.lower()


class ExtensionInCondition(ICondition):
    """Condition: file extension in list."""
    
    def __init__(self, extensions: list):
        self.extensions = [e.lower() for e in extensions]
    
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        return file.extension.lower() in self.extensions


class TagHasCondition(ICondition):
    """Condition: file has specific tag."""
    
    def __init__(self, tag_id: str):
        self.tag_id = tag_id
    
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        from bson import ObjectId
        return ObjectId(self.tag_id) in file.tag_ids


class RatingGteCondition(ICondition):
    """Condition: rating >= threshold."""
    
    def __init__(self, threshold: int):
        self.threshold = threshold
    
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        return file.rating >= self.threshold


class SizeGteCondition(ICondition):
    """Condition: file size >= threshold."""
    
    def __init__(self, threshold: int):
        self.threshold = threshold
    
    def evaluate(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        return file.size_bytes >= self.threshold


# Condition registry
CONDITION_TYPES = {
    "path_contains": PathContainsCondition,
    "extension_in": ExtensionInCondition,
    "tag_has": TagHasCondition,
    "rating_gte": RatingGteCondition,
    "size_gte": SizeGteCondition,
}


def create_condition(condition_type: str, params: Dict[str, Any]) -> ICondition:
    """
    Create condition from type and params.
    
    Args:
        condition_type: Type identifier
        params: Condition parameters
        
    Returns:
        ICondition instance
    """
    condition_class = CONDITION_TYPES.get(condition_type)
    if not condition_class:
        logger.warning(f"Unknown condition type: {condition_type}")
        return None
    
    return condition_class(**params)
