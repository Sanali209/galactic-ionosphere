"""
Pytest configuration for UExplorer tests.
"""
import pytest
import asyncio
from pathlib import Path

# Add Foundation to path
# Now import after path is set
from src.core.bootstrap import ApplicationBuilder
from src.core.services.fs_service import FSService
from src.ucorefs.discovery.service import DiscoveryService
from src.ucorefs.thumbnails.service import ThumbnailService
from src.ucorefs.vectors.service import VectorService
from src.ucorefs.ai.similarity_service import SimilarityService
from src.ucorefs.ai.llm_service import LLMService
from src.ucorefs.relations.service import RelationService
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.albums.manager import AlbumManager
from src.ucorefs.rules.engine import RulesEngine

@pytest.fixture(scope="session")
def qapp(qapp_args):
    """Qt Application for all tests."""
    from PySide6.QtWidgets import QApplication
    app = QApplication(qapp_args)
    yield app
    app.quit()

@pytest.fixture(scope="session")
def event_loop():
    """Create and set event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def locator(event_loop):
    """Provide test ServiceLocator with all UCoreFS systems."""
    config_path = Path(__file__).parent.parent / "config.toml"
    
    async def build():
        builder = (ApplicationBuilder("UExplorer Test", str(config_path))
                  .with_default_systems()
                  .add_system(FSService)
                  .add_system(DiscoveryService)
                  .add_system(ThumbnailService)
                  .add_system(VectorService)
                  .add_system(SimilarityService)
                  .add_system(LLMService)
                  .add_system(RelationService)
                  .add_system(TagManager)
                  .add_system(AlbumManager)
                  .add_system(RulesEngine))
        
        return await builder.build()
    
    locator_instance = event_loop.run_until_complete(build())
    yield locator_instance
    
    # Cleanup
    async def cleanup():
        await locator_instance.stop_all()
    event_loop.run_until_complete(cleanup())
