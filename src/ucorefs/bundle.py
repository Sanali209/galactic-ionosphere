"""
UCoreFS System Bundle.

Groups all UCoreFS services for easy registration with ApplicationBuilder.
"""
from src.core.bootstrap import SystemBundle

if False:  # TYPE_CHECKING
    from src.core.bootstrap import ApplicationBuilder


class UCoreFSEngineBundle(SystemBundle):
    """
    Bundle containing ALL services for the Background Engine.
    Includes TaskSystem, AI, Processing, and all Data services.
    """
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """Register all services for the Engine."""
        # Core services
        from src.ucorefs.services.fs_service import FSService
        from src.ucorefs.services.maintenance_service import MaintenanceService
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        from src.ucorefs.discovery.service import DiscoveryService
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        builder.add_system(FSService)
        builder.add_system(MaintenanceService)
        builder.add_system(ProcessingPipeline)
        builder.add_system(DiscoveryService)
        builder.add_system(ThumbnailService)
        
        # Vector/AI services
        from src.ucorefs.vectors.faiss_service import FAISSIndexService
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.search.service import SearchService
        from src.ucorefs.ai.similarity_service import SimilarityService
        from src.ucorefs.ai.llm_service import LLMService
        from src.core.llm.worker_service import LLMWorkerService
        from src.core.tasks.system import TaskSystem
        
        builder.add_system(TaskSystem)  # TaskSystem owns AI executor
        builder.add_system(FAISSIndexService)
        builder.add_system(VectorService)
        builder.add_system(SearchService)
        builder.add_system(SimilarityService)
        builder.add_system(LLMService)
        builder.add_system(LLMWorkerService)
        
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
        
        builder.add_system(DetectionService)
        builder.add_system(WDTaggerService)
        builder.add_system(AnnotationService)
        
        # AI Extractors
        from src.ucorefs.extractors.clip_extractor import CLIPExtractor
        from src.ucorefs.extractors.blip_extractor import BLIPExtractor
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        builder.add_system(CLIPExtractor)
        builder.add_system(BLIPExtractor)
        builder.add_system(GroundingDINOExtractor)


class UCoreFSClientBundle(SystemBundle):
    """
    Bundle containing subset of services for the UI Client.
    Read-only access to Data services. No heavy processing.
    """
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """Register UI-facing services."""
        # Core Data Access (Read)
        from src.ucorefs.services.fs_service import FSService
        from src.ucorefs.search.service import SearchService
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        builder.add_system(FSService)
        builder.add_system(SearchService)
        builder.add_system(ThumbnailService)
        
        # Dependencies of Search/Thumbnail might include vectors
        from src.ucorefs.vectors.faiss_service import FAISSIndexService
        from src.ucorefs.vectors.service import VectorService
        
        builder.add_system(FAISSIndexService) # Read-only index mirror
        builder.add_system(VectorService)
        
        # Organization (Read/Display)
        from src.ucorefs.relations.service import RelationService
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.albums.manager import AlbumManager
        
        builder.add_system(RelationService)
        builder.add_system(TagManager)
        builder.add_system(AlbumManager)
        
        # Note: TaskSystem, ProcessingPipeline, AI Models are EXCLUDED.


# Backward compatibility
class UCoreFSBundle(UCoreFSEngineBundle):
    pass
