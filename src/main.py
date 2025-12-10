import sys
import asyncio
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl, QObject
from loguru import logger

from src.core.locator import sl
from src.core.database.manager import db_manager
from src.core.engine.importer import ImportService
from src.core.engine.tasks import TaskDispatcher
from src.core.engine.monitor import FileMonitor
from src.core.ai.vector_driver import VectorDriver
from src.core.ai.embeddings import EmbeddingService
from src.core.ai.detection import ObjectDetectionService
from src.core.ai.search import SearchService
from src.core.ai.handlers import AITaskHandlers

from src.ui.models.navigation import FolderTreeModel, TagTreeModel
from src.ui.models.grid_model import GalleryGridModel
from src.ui.bridge import BackendBridge

async def main():
    # 0. Initialize Service Locator
    sl.init("config.yaml")

    # 1. Initialize DB
    # init() is synchronous but ensures connection client is ready
    db_manager.init()
    
    # 2. Initialize Core Services
    # (In a real DI container this would be cleaner)
    tasks = TaskDispatcher()
    await tasks.start()
    
    importer = ImportService(dispatcher=tasks)
    
    vector_driver = VectorDriver("localhost", 6333)
    # Check connection (non-blocking in UI? For now await)
    try:
        vector_driver.connect()
    except Exception:
        logger.warning("Vector DB not connected")

    embed_service = EmbeddingService()
    # Lazy load embed service usually
    
    search_service = SearchService(vector_driver, embed_service)
    
    detect_service = ObjectDetectionService()
    # detect_service.load() # Lazy loading preferred
    
    ai_handlers = AITaskHandlers(vector_driver, embed_service, detect_service)
    tasks.register_handler("GENERATE_VECTORS", ai_handlers.handle_generate_vectors)
    tasks.register_handler("DETECT_OBJECTS", ai_handlers.handle_detect_objects)
    
    # Monitor (Optional start)
    # monitor = FileMonitor(importer.process_file)
    # monitor.start()

    # 3. Initialize UI Models
    folder_model = FolderTreeModel()
    tag_model = TagTreeModel()
    tag_model.load_tags() # Start Async Load
    
    grid_model = GalleryGridModel()
    
    bridge = BackendBridge(importer, search_service, grid_model)

    # 4. Setup Qt / Hybrid
    # Set Style to Fusion to support customization
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    
    # Check for PySide6-QtAds if we want to upgrade later, but for now standard docks
    
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    from src.ui.main_window import MainWindow
    window = MainWindow(bridge, folder_model, tag_model, grid_model)
    window.show()
    
    logger.info("Application Started (Hybrid Mode)")
    
    # Keep reference to window to prevent GC
    engine = window 
    
    # Keep running via loop
    # Return execution to the caller (qasync loop_forever)
    return engine

if __name__ == "__main__":
    # We need qasync to combine asyncio and Qt event loop
    try:
        import qasync
    except ImportError:
        logger.error("qasync is required. pip install qasync")
        sys.exit(1)

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    with loop:
        # main() is now async
        main_window_ref = loop.run_until_complete(main())
        loop.run_forever()

