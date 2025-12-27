"""
UCoreFS - Filesystem Database System

A MongoDB-backed filesystem database with AI capabilities.
Built on the Foundation template architecture.

Features:
- Hierarchical file/directory records with virtual file support
- Background discovery with watch/blacklists
- Multi-phase processing pipeline (Phase 1-3)
- Polymorphic file types with driver system
- Thumbnail caching and vector search (FAISS)
- Detection instances (bounding boxes as virtual files)
- Relation system (duplicate detection, etc.)
- Hierarchical tags with synonyms/antonyms
- Smart albums with dynamic queries
- Rules engine for automation
- Powerful query builder with AND/OR/NOT

Note: Uses lazy imports to avoid circular dependencies.
Direct imports: from src.ucorefs.models import FileRecord
"""

__version__ = "0.2.0"

# Lazy import system to avoid circular dependencies
_lazy_imports = {
    # Models
    "FSRecord": "src.ucorefs.models.base",
    "FileRecord": "src.ucorefs.models.file_record",
    "DirectoryRecord": "src.ucorefs.models.directory",
    "ProcessingState": "src.ucorefs.models.base",
    
    # Services
    "FSService": "src.ucorefs.services.fs_service",
    "DiscoveryService": "src.ucorefs.discovery.service",
    "ThumbnailService": "src.ucorefs.thumbnails.service",
    "VectorService": "src.ucorefs.vectors.service",
    "SimilarityService": "src.ucorefs.ai.similarity_service",
    "LLMService": "src.ucorefs.ai.llm_service",
    "ProcessingPipeline": "src.ucorefs.processing.pipeline",
    
    # File types
    "IFileDriver": "src.ucorefs.types.driver",
    "FileTypeRegistry": "src.ucorefs.types.registry",
    
    # Detection & Relations
    "DetectionClass": "src.ucorefs.detection.models",
    "DetectionObject": "src.ucorefs.detection.models",
    "DetectionInstance": "src.ucorefs.detection.models",
    "Relation": "src.ucorefs.relations.models",
    "RelationType": "src.ucorefs.relations.models",
    "RelationService": "src.ucorefs.relations.service",
    
    # Tags & Albums
    "Tag": "src.ucorefs.tags.model",
    "TagManager": "src.ucorefs.tags.manager",
    "Album": "src.ucorefs.albums.model",
    "AlbumManager": "src.ucorefs.albums.manager",
    
    # Rules
    "Rule": "src.ucorefs.rules.model",
    "RulesEngine": "src.ucorefs.rules.engine",
    "ICondition": "src.ucorefs.rules.conditions",
    "IAction": "src.ucorefs.rules.actions",
    
    # Query
    "QueryBuilder": "src.ucorefs.query.builder",
    "Q": "src.ucorefs.query.builder",
    "Aggregation": "src.ucorefs.query.aggregation",
}

__all__ = list(_lazy_imports.keys())

def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module_path = _lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
