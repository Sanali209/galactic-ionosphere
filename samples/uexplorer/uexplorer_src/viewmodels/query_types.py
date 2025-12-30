"""
Query reference types for lightweight entity storage.

These types provide a middle ground between storing just IDs (current approach)
and full ORM objects, containing only the essential fields needed for UI display
and query building.
"""
from dataclasses import dataclass
from bson import ObjectId
from typing import Optional


@dataclass
class TagRef:
    """
    Lightweight tag reference for queries.
    
    Stores essential tag information needed for query building and UI display
    without requiring a full database query.
    
    Attributes:
        id: MongoDB ObjectId of the tag
        name: Display name (e.g., "sky", "cat")
        full_path: Hierarchical path (e.g., "Animals/Mammals/Cats")
        color: UI color for the tag (optional)
    """
    id: ObjectId
    name: str
    full_path: str = ""
    color: str = ""
    
    def to_id_str(self) -> str:
        """
        Convert to string ID for MongoDB queries.
        
        Returns:
            String representation of ObjectId
        """
        return str(self.id)
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON/session storage.
        
        Returns:
            Dictionary with serializable values
        """
        return {
            'id': str(self.id),
            'name': self.name,
            'full_path': self.full_path,
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TagRef':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary from to_dict() or JSON
            
        Returns:
            TagRef instance
        """
        return cls(
            id=ObjectId(data['id']),
            name=data['name'],
            full_path=data.get('full_path', ''),
            color=data.get('color', '')
        )
    
    def __str__(self) -> str:
        return f"TagRef({self.name})"
    
    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Equality based on ID."""
        if not isinstance(other, TagRef):
            return False
        return self.id == other.id


@dataclass
class AlbumRef:
    """
    Lightweight album reference for queries.
    
    Stores essential album information needed for query building and UI display.
    
    Attributes:
        id: MongoDB ObjectId of the album
        name: Display name (e.g., "Vacation 2024")
        description: Album description (optional)
        is_smart: Whether this is a smart album (dynamic query)
    """
    id: ObjectId
    name: str
    description: str = ""
    is_smart: bool = False
    
    def to_id_str(self) -> str:
        """
        Convert to string ID for MongoDB queries.
        
        Returns:
            String representation of ObjectId
        """
        return str(self.id)
    
    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON/session storage.
        
        Returns:
            Dictionary with serializable values
        """
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'is_smart': self.is_smart
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AlbumRef':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary from to_dict() or JSON
            
        Returns:
            AlbumRef instance
        """
        return cls(
            id=ObjectId(data['id']),
            name=data['name'],
            description=data.get('description', ''),
            is_smart=data.get('is_smart', False)
        )
    
    def __str__(self) -> str:
        smart_indicator = " [Smart]" if self.is_smart else ""
        return f"AlbumRef({self.name}{smart_indicator})"
    
    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Equality based on ID."""
        if not isinstance(other, AlbumRef):
            return False
        return self.id == other.id
