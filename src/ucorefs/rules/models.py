"""
UCoreFS - Rule Models

Models for automation rules.
"""
from typing import List, Dict, Any
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class Rule(CollectionRecord):
    """
    Automation rule with conditions and actions.
    
    Executes actions when conditions are met on trigger.
    Auto collection name: "rules"
    """
    # Rule info
    name: str = Field(default="", index=True)
    description: str = Field(default="")
    enabled: bool = Field(default=True, index=True)
    
    # Trigger
    trigger: str = Field(default="manual", index=True)  # on_import, on_tag, manual
    
    # Conditions (must all match)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Actions (executed in order)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Statistics
    execution_count: int = Field(default=0)
    last_executed: str = Field(default="")
    
    def __str__(self) -> str:
        return f"Rule: {self.name} ({self.trigger}, {len(self.conditions)} conditions)"
