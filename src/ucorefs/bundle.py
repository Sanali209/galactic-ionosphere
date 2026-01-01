"""
UCoreFS System Bundle.

Groups all UCoreFS services for easy registration with ApplicationBuilder.
"""
from src.core.bootstrap import SystemBundle

if False:  # TYPE_CHECKING
    from src.core.bootstrap import ApplicationBuilder


class UCoreFSBundle(SystemBundle):
    """
    Bundle containing all UCoreFS services in correct dependency order.
    
    Encapsulates file management, AI processing, search, and organization
    services into a single reusable bundle.
    
    Example:
        builder = (ApplicationBuilder("MyApp", "config.json")
                   .with_default_systems()
                   .add_bundle(UCoreFSBundle()))
    """
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """Register all UCoreFS services in dependency order."""
        # Core services
        from src.ucorefs.services.fs_service import FSService
        from src.ucorefs.services.maintenance_service import MaintenanceService
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        from src.ucorefs.discovery.service import DiscoveryService
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        builder.add_system(FSService)
        builder.add_system(MaintenanceService)  # Data integrity & count management
        builder.add_system(ProcessingPipeline)  # Must be before DiscoveryService
        builder.add_system(DiscoveryService)
        builder.add_system(ThumbnailService)
        
        # Vector/AI services (order matters: FAISS → Vector → Search)
        from src.ucorefs.vectors.faiss_service import FAISSIndexService
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.search.service import SearchService
        from src.ucorefs.ai.similarity_service import SimilarityService
        from src.ucorefs.ai.llm_service import LLMService
        
        builder.add_system(FAISSIndexService)   # Must be before VectorService
        builder.add_system(VectorService)
        builder.add_system(SearchService)       # Unified search (uses FAISS + MongoDB)
        builder.add_system(SimilarityService)
        builder.add_system(LLMService)
        
        # Organization services
        from src.ucorefs.relations.service import RelationService
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.albums.manager import AlbumManager
        from src.ucorefs.rules.engine import RulesEngine
        
        builder.add_system(RelationService)
        builder.add_system(TagManager)
        builder.add_system(AlbumManager)
        builder.add_system(RulesEngine)
        
        # AI detection and annotation services
        from src.ucorefs.detection import DetectionService
        from src.ucorefs.services.wd_tagger_service import WDTaggerService
        from src.ucorefs.annotation.service import AnnotationService
        
        builder.add_system(DetectionService)    # Object detection (YOLO/MTCNN)
        builder.add_system(WDTaggerService)     # AI image tagging
        builder.add_system(AnnotationService)
