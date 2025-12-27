"""
FieldRegistry - Extensible field definitions for search and filters.

Defines searchable/filterable fields with their properties,
enabling auto-generation of filter UI elements.
"""
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class FieldType(Enum):
    """Types of fields."""
    TEXT = "text"           # Free text search
    SELECT = "select"       # Single select from options
    MULTI_SELECT = "multi"  # Multi-select from options
    RANGE = "range"         # Numeric range (min-max)
    BOOLEAN = "boolean"     # True/False toggle
    DATE = "date"           # Date range
    TAGS = "tags"           # Tag IDs


class Operator(Enum):
    """Filter operators for building queries."""
    EQUALS = "eq"           # Exact match
    NOT_EQUALS = "ne"       # Not equal
    CONTAINS = "contains"   # Text contains
    STARTS_WITH = "starts"  # Starts with
    ENDS_WITH = "ends"      # Ends with
    GREATER_THAN = "gt"     # Greater than
    GREATER_EQ = "gte"      # Greater or equal
    LESS_THAN = "lt"        # Less than
    LESS_EQ = "lte"         # Less or equal
    IN = "in"               # In list
    NOT_IN = "nin"          # Not in list
    ALL = "all"             # Has all (for arrays)
    REGEX = "regex"         # Regex match
    EXISTS = "exists"       # Field exists
    BETWEEN = "between"     # Between min/max


# Default operators by field type
DEFAULT_OPERATORS: Dict[FieldType, List[Operator]] = {
    FieldType.TEXT: [Operator.CONTAINS, Operator.EQUALS, Operator.STARTS_WITH, Operator.REGEX],
    FieldType.SELECT: [Operator.EQUALS, Operator.NOT_EQUALS],
    FieldType.MULTI_SELECT: [Operator.IN, Operator.NOT_IN, Operator.ALL],
    FieldType.RANGE: [Operator.EQUALS, Operator.GREATER_THAN, Operator.LESS_THAN, Operator.BETWEEN],
    FieldType.BOOLEAN: [Operator.EQUALS],
    FieldType.DATE: [Operator.EQUALS, Operator.GREATER_THAN, Operator.LESS_THAN, Operator.BETWEEN],
    FieldType.TAGS: [Operator.ALL, Operator.IN, Operator.NOT_IN],
}


@dataclass
class FieldDefinition:
    """
    Definition of a searchable/filterable field.
    
    Attributes:
        name: Internal field name (matches DB field)
        label: Display label
        field_type: Type of field (affects UI)
        searchable: Can be searched via text
        filterable: Can be filtered
        options: For SELECT/MULTI_SELECT types
        default: Default value
        db_path: MongoDB path (e.g., "metadata.width")
        operators: Allowed filter operators (auto-set from type)
        editor_factory: Callable that creates editor widget
    """
    name: str
    label: str
    field_type: FieldType = FieldType.TEXT
    searchable: bool = True
    filterable: bool = True
    options: Optional[List[tuple]] = None  # [(value, label), ...]
    default: Any = None
    db_path: Optional[str] = None  # Custom DB path, defaults to name
    operators: Optional[List[Operator]] = None  # Custom operators, or use defaults
    editor_factory: Optional[Callable] = None  # Factory function for editor widget
    
    def __post_init__(self):
        """Set default operators if not provided."""
        if self.operators is None:
            self.operators = DEFAULT_OPERATORS.get(self.field_type, [Operator.EQUALS])
    
    @property
    def mongo_path(self) -> str:
        """Get MongoDB field path."""
        return self.db_path or self.name
    
    def get_operators(self) -> List[Operator]:
        """Get available operators for this field."""
        return self.operators or []


class FieldRegistry:
    """
    Registry of all searchable/filterable fields.
    
    Usage:
        registry = FieldRegistry()
        registry.register_defaults()
        
        # Get all searchable fields
        for field in registry.get_searchable_fields():
            print(field.label)
        
        # Build filter UI dynamically
        for field in registry.get_filterable_fields():
            create_filter_widget(field)
    """
    
    def __init__(self):
        self._fields: Dict[str, FieldDefinition] = {}
        logger.debug("FieldRegistry initialized")
    
    def register(self, field_def: FieldDefinition):
        """Register a field definition."""
        self._fields[field_def.name] = field_def
        logger.debug(f"Registered field: {field_def.name}")
    
    def register_defaults(self):
        """Register default UExplorer fields."""
        
        # Core fields
        self.register(FieldDefinition(
            name="name",
            label="Name",
            field_type=FieldType.TEXT,
            searchable=True,
            filterable=False
        ))
        
        self.register(FieldDefinition(
            name="path",
            label="Path",
            field_type=FieldType.TEXT,
            searchable=True,
            filterable=False
        ))
        
        self.register(FieldDefinition(
            name="extension",
            label="Extension",
            field_type=FieldType.TEXT,
            searchable=True,
            filterable=True
        ))
        
        # File type
        self.register(FieldDefinition(
            name="file_type",
            label="File Type",
            field_type=FieldType.MULTI_SELECT,
            searchable=False,
            filterable=True,
            options=[
                ("image", "Images"),
                ("video", "Videos"),
                ("audio", "Audio"),
                ("document", "Documents"),
                ("archive", "Archives"),
            ]
        ))
        
        # Rating
        self.register(FieldDefinition(
            name="rating",
            label="Rating",
            field_type=FieldType.RANGE,
            searchable=False,
            filterable=True,
            default=0
        ))
        
        # Size
        self.register(FieldDefinition(
            name="size",
            label="File Size",
            field_type=FieldType.RANGE,
            searchable=False,
            filterable=True
        ))
        
        # Tags
        self.register(FieldDefinition(
            name="tag_ids",
            label="Tags",
            field_type=FieldType.TAGS,
            searchable=True,
            filterable=True
        ))
        
        # AI Description
        self.register(FieldDefinition(
            name="ai_description",
            label="AI Description",
            field_type=FieldType.TEXT,
            searchable=True,
            filterable=False
        ))
        
        # Description
        self.register(FieldDefinition(
            name="description",
            label="Description",
            field_type=FieldType.TEXT,
            searchable=True,
            filterable=False
        ))
        
        # Image dimensions
        self.register(FieldDefinition(
            name="width",
            label="Width",
            field_type=FieldType.RANGE,
            searchable=False,
            filterable=True,
            db_path="metadata.width"
        ))
        
        self.register(FieldDefinition(
            name="height",
            label="Height",
            field_type=FieldType.RANGE,
            searchable=False,
            filterable=True,
            db_path="metadata.height"
        ))
        
        logger.info(f"Registered {len(self._fields)} default fields")
    
    def get(self, name: str) -> Optional[FieldDefinition]:
        """Get field by name."""
        return self._fields.get(name)
    
    def get_all(self) -> List[FieldDefinition]:
        """Get all registered fields."""
        return list(self._fields.values())
    
    def get_searchable_fields(self) -> List[FieldDefinition]:
        """Get fields that support text search."""
        return [f for f in self._fields.values() if f.searchable]
    
    def get_filterable_fields(self) -> List[FieldDefinition]:
        """Get fields that support filtering."""
        return [f for f in self._fields.values() if f.filterable]
    
    def get_by_type(self, field_type: FieldType) -> List[FieldDefinition]:
        """Get fields by type."""
        return [f for f in self._fields.values() if f.field_type == field_type]


# Singleton instance
_registry: Optional[FieldRegistry] = None


def get_field_registry() -> FieldRegistry:
    """Get the global FieldRegistry instance."""
    global _registry
    if _registry is None:
        _registry = FieldRegistry()
        _registry.register_defaults()
    return _registry
