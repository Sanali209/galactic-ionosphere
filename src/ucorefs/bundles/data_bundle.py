"""
UCoreFS Data Layer Bundle.

Framework-agnostic bundle that registers data access services.
Works in both console and GUI applications.

Usage:
    # Console application
    builder = (ApplicationBuilder.for_console("CLI", "config.json")
        .add_bundle(UCoreFSDataBundle()))
    
    # GUI application
    builder = (ApplicationBuilder.for_gui("UExplorer", "config.json")
        .add_bundle(UCoreFSDataBundle())
        .add_bundle(PySideBundle()))
"""
from typing import TYPE_CHECKING

from src.core.bootstrap import SystemBundle

if TYPE_CHECKING:
    from src.core.bootstrap import ApplicationBuilder


class UCoreFSDataBundle(SystemBundle):
    """
    UCoreFS data access services (framework-agnostic).
    
    Works in console AND GUI applications.
    Provides read/write access to:
    - File system metadata
    - Tags & Albums
    - Relations
    - Search queries
    - Vector embeddings
    
    Does NOT include:
    - AI processing (see UCoreFSEngineBundle)
    - UI framework (see PySideBundle)
    
    Example:
        # Read-write mode (default)
        bundle = UCoreFSDataBundle()
        
        # Read-only mode (for viewers/queries)
        bundle = UCoreFSDataBundle(readonly=True)
    """
    
    def __init__(self, readonly: bool = False):
        """
        Initialize data bundle.
        
        Args:
            readonly: If True, skip write-dependent services like DiscoveryService
        """
        self.readonly = readonly
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """
        Register data layer services.
        
        Args:
            builder: ApplicationBuilder to register systems with
        """
        # Core data services
        from src.ucorefs.services.fs_service import FSService
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.albums.manager import AlbumManager
        from src.ucorefs.relations.service import RelationService
        from src.ucorefs.search.service import SearchService
        
        builder.add_system(FSService)
        builder.add_system(TagManager)
        builder.add_system(AlbumManager)
        builder.add_system(RelationService)
        builder.add_system(SearchService)
        
        # Vector services (needed for search)
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.vectors.faiss_service import FAISSIndexService
        
        builder.add_system(VectorService)
        builder.add_system(FAISSIndexService)
        
        # Thumbnail service (display)
        from src.ucorefs.thumbnails.service import ThumbnailService
        builder.add_system(ThumbnailService)
        
        # Write-dependent services (only if not readonly)
        if not self.readonly:
            from src.ucorefs.discovery.service import DiscoveryService
            builder.add_system(DiscoveryService)
