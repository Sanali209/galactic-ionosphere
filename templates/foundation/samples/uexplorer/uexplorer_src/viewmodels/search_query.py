"""
SearchQuery - Data class for search parameters.

Encapsulates all search options for the SearchPipeline.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from bson import ObjectId


@dataclass
class SearchQuery:
    """
    Search query parameters.
    
    Attributes:
        text: Search text
        mode: "text", "vector", or "image"
        fields: Fields to search (for text mode)
        file_id: File ID for image similarity search
        filters: Filter conditions from FilterPanel
        tags: Selected tag IDs
        tag_mode: "any", "all", "none"
        directory: Current directory (for scoping)
        limit: Max results
    """
    text: str = ""
    mode: str = "text"  # text | vector | image
    fields: List[str] = field(default_factory=lambda: ["name"])
    file_id: Optional[ObjectId] = None  # For image similarity
    filters: Dict[str, Any] = field(default_factory=dict)
    tags: List[ObjectId] = field(default_factory=list)
    tag_mode: str = "any"
    directory: Optional[ObjectId] = None
    limit: int = 100
    
    def is_empty(self) -> bool:
        """Check if query has no search criteria."""
        return (
            not self.text 
            and not self.filters 
            and not self.tags
            and not self.file_id
        )
    
    def is_text_search(self) -> bool:
        """Check if this is a text search."""
        return self.mode == "text" and bool(self.text)
    
    def is_vector_search(self) -> bool:
        """Check if this is a vector (text→embedding) search."""
        return self.mode == "vector" and bool(self.text)
    
    def is_image_search(self) -> bool:
        """Check if this is an image similarity search."""
        return self.mode == "image" and self.file_id is not None
