"""
UExplorer ViewModels - MVVM state management.

NOTE: UExplorer uses its own DocumentManager that is QObject-based for
widget parenting. Foundation DocumentManager is available separately.
"""
from uexplorer_src.viewmodels.browse_view_model import BrowseViewModel
from uexplorer_src.viewmodels.document_manager import DocumentManager
from uexplorer_src.viewmodels.search_query import SearchQuery
from uexplorer_src.viewmodels.search_pipeline import SearchPipeline
from uexplorer_src.viewmodels.field_registry import (
    FieldRegistry, FieldDefinition, FieldType, get_field_registry
)

__all__ = [
    "BrowseViewModel",
    "DocumentManager", 
    "SearchQuery",
    "SearchPipeline",
    "FieldRegistry",
    "FieldDefinition",
    "FieldType",
    "get_field_registry",
]


