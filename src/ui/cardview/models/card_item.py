"""
CardItem Data Model.

Pydantic model for card data with all required fields
for display, sorting, filtering, and grouping.
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SortOrder(str, Enum):
    """Sort direction."""
    ASCENDING = "asc"
    DESCENDING = "desc"


class FilterOperator(str, Enum):
    """Logical operator for combining filters."""
    AND = "and"
    OR = "or"


class CardItem(BaseModel):
    """
    Data model for a single card item.
    
    Attributes:
        id: Unique identifier (required for virtualization)
        title: Primary display text
        subtitle: Secondary text (optional)
        thumbnail_path: Path to thumbnail image (optional)
        item_type: Type for template selection (default: "default")
        data: Original data object for custom processing
        group_key: Key for grouping (optional)
        sort_key: Custom sort value (optional)
        rating: Rating value 0-5 (optional)
        tags: List of tag strings (optional)
    
    Example:
        item = CardItem(
            id="file_123",
            title="vacation.jpg",
            subtitle="2.4 MB",
            thumbnail_path="/path/to/file.jpg",
            item_type="image",
            data=file_record
        )
    """
    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Primary display text")
    subtitle: Optional[str] = Field(None, description="Secondary text")
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail")
    item_type: str = Field("default", description="Type for template selection")
    data: Any = Field(None, description="Original data object")
    group_key: Optional[str] = Field(None, description="Grouping key")
    sort_key: Optional[Any] = Field(None, description="Custom sort value")
    rating: int = Field(0, ge=0, le=5, description="Rating 0-5")
    tags: list[str] = Field(default_factory=list, description="Tag strings")
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    def get_field(self, field_name: str) -> Any:
        """
        Get field value by name for dynamic sorting/grouping.
        
        Args:
            field_name: Name of field to retrieve
            
        Returns:
            Field value or None if not found
        """
        if hasattr(self, field_name):
            return getattr(self, field_name)
        if self.data and hasattr(self.data, field_name):
            return getattr(self.data, field_name)
        return None
    
    def matches_text(self, text: str, fields: list[str] | None = None) -> bool:
        """
        Check if item matches search text.
        
        Args:
            text: Search text (case-insensitive)
            fields: Fields to search (default: title, subtitle, tags)
            
        Returns:
            True if text found in any field
        """
        if not text:
            return True
        
        text_lower = text.lower()
        search_fields = fields or ["title", "subtitle", "tags"]
        
        for field in search_fields:
            value = self.get_field(field)
            if value is None:
                continue
            if isinstance(value, str) and text_lower in value.lower():
                return True
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and text_lower in item.lower():
                        return True
        
        return False
