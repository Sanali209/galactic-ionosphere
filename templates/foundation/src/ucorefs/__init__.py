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
"""

# Core models
from src.ucorefs.models import FSRecord, FileRecord, DirectoryRecord, ProcessingState

# Services
from src.ucorefs.core import FSService
from src.ucorefs.discovery import DiscoveryService
from src.ucorefs.thumbnails import ThumbnailService
from src.ucorefs.vectors import VectorService
from src.ucorefs.ai import SimilarityService, LLMService
from src.ucorefs.processing import ProcessingPipeline

# File types
from src.ucorefs.types import IFileDriver, FileTypeRegistry, registry

# Detection & Relations
from src.ucorefs.detection import DetectionClass, DetectionObject, DetectionInstance
from src.ucorefs.relations import Relation, RelationType, RelationService

# Tags & Albums
from src.ucorefs.tags import Tag, TagManager
from src.ucorefs.albums import Album, AlbumManager

# Rules
from src.ucorefs.rules import Rule, RulesEngine, ICondition, IAction

# Query
from src.ucorefs.query import QueryBuilder, Q, Aggregation

__version__ = "0.2.0"

__all__ = [
    # Models
    "FSRecord",
    "FileRecord",
    "DirectoryRecord",
    "ProcessingState",
    "DetectionClass",
    "DetectionObject",
    "DetectionInstance",
    "Relation",
    "RelationType",
    "Tag",
    "Album",
    "Rule",
    
    # Services
    "FSService",
    "DiscoveryService",
    "ThumbnailService",
    "VectorService",
    "SimilarityService",
    "LLMService",
    "RelationService",
    "TagManager",
    "AlbumManager",
    "RulesEngine",
    "ProcessingPipeline",
    
    # File Types
    "IFileDriver",
    "FileTypeRegistry",
    "registry",
    
    # Rules
    "ICondition",
    "IAction",
    
    # Query
    "QueryBuilder",
    "Q",
    "Aggregation",
]

