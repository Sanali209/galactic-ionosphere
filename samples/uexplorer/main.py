"""
UExplorer - Main Entry Point

Directory Opus-inspired file manager showcasing Foundation + UCoreFS.

Refactored to use Foundation's run_app() helper for cleaner startup.
"""
import sys
from pathlib import Path

# Add foundation to path (allows running main.py directly for debugging)
foundation_path = Path(__file__).parent.parent.parent
if str(foundation_path) not in sys.path:
    sys.path.insert(0, str(foundation_path))

# Add uexplorer to path
uexplorer_path = Path(__file__).parent
if str(uexplorer_path) not in sys.path:
    sys.path.insert(0, str(uexplorer_path))

from loguru import logger

# Import Foundation bootstrap
from src.core.bootstrap import ApplicationBuilder, run_app

# Import UCoreFS systems
from src.ucorefs.services.fs_service import FSService
from src.ucorefs.processing.pipeline import ProcessingPipeline
from src.ucorefs.discovery.service import DiscoveryService
from src.ucorefs.thumbnails.service import ThumbnailService
from src.ucorefs.vectors.service import VectorService
from src.ucorefs.search.service import SearchService
from src.ucorefs.ai.similarity_service import SimilarityService
from src.ucorefs.ai.llm_service import LLMService
from src.ucorefs.relations.service import RelationService
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.albums.manager import AlbumManager
from src.ucorefs.rules.engine import RulesEngine
from src.ucorefs.detection import DetectionService
from src.ucorefs.annotation.service import AnnotationService
from src.ucorefs.services.wd_tagger_service import WDTaggerService

# Import UExplorer UI
from uexplorer_src.ui.main_window import MainWindow
from uexplorer_src.ui.viewmodels.main_viewmodel import MainViewModel


def main():
    """Main entry point using Foundation's run_app helper."""
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Starting")
    logger.info("=" * 60)
    
    config_path = Path(__file__).parent / "config.json"
    
    # Import FAISS service and SessionState
    from src.ucorefs.vectors.faiss_service import FAISSIndexService
    from src.ui.state import SessionState
    from src.ui.navigation.service import NavigationService
    
    # Build application with all UCoreFS systems
    # Order: FAISS → VectorService → SearchService
    builder = (
        ApplicationBuilder("UExplorer", str(config_path))
        .with_default_systems()
        .with_logging(True)
        .add_system(SessionState)        # UI session persistence
        .add_system(NavigationService)   # Smart Selection Routing
        .add_system(FSService)
        .add_system(ProcessingPipeline)  # Must be before DiscoveryService
        .add_system(DiscoveryService)
        .add_system(ThumbnailService)
        .add_system(FAISSIndexService)   # Must be before VectorService
        .add_system(VectorService)
        .add_system(SearchService)       # Unified search (uses FAISS + MongoDB)
        .add_system(SimilarityService)
        .add_system(LLMService)
        .add_system(RelationService)
        .add_system(TagManager)
        .add_system(AlbumManager)
        .add_system(RulesEngine)
        .add_system(DetectionService)    # Object detection (YOLO/MTCNN)
        .add_system(WDTaggerService)     # AI image tagging
        .add_system(AnnotationService)
    )
    
    # Run with Foundation's helper (handles Qt, async, shutdown)
    run_app(MainWindow, MainViewModel, builder=builder)
    
    logger.info("👋 UExplorer shutdown complete")


if __name__ == "__main__":
    main()
