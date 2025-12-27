"""
UCoreFS Services.

Provides the main service layer for filesystem database operations.

FSService is the primary entry point for:
- Managing library roots
- Navigating directory hierarchy
- CRUD operations on files and directories

WDTaggerService provides AI-powered image tagging.
"""
from .fs_service import FSService
from .wd_tagger_service import WDTaggerService

__all__ = ["FSService", "WDTaggerService"]


