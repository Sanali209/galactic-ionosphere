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
from src.core.ai.vector_driver import VectorDriver
from src.core.ai.service import EmbeddingService
from src.core.ai.detection import ObjectDetectionService
from src.core.ai.search import SearchService
from src.core.ai.handlers import AITaskHandlers

from src.ui.models.navigation import FolderTreeModel, TagTreeModel
from src.ui.models.grid_model import GalleryGridModel
from src.ui.bridge import BackendBridge

async def main():
    # 0. Initialize Service Locator
    sl.init("config.json")

    # 1. Initialize DB
    # init() is synchronous but ensures connection client is ready
    db_manager.init()
    
    # Start Journal Service (Needs DB)
    if sl.journal:
        sl.journal.start()
    
    # 2. Initialize Core Services
    # (In a real DI container this would be cleaner)
    # 2. Initialize Core Services
    tasks = TaskDispatcher()
    await tasks.start()
    
    vector_driver = VectorDriver("localhost", 6333)
    try:
        vector_driver.connect()
    except Exception:
        logger.warning("Vector DB not connected")

    embed_service = EmbeddingService()
    
    importer = ImportService(dispatcher=tasks, embed_service=embed_service, vector_driver=vector_driver)
    
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
    # folder_model = FolderTreeModel() # Legacy
    
    from src.ui.models.flat_tree import FileSystemFlatModel, TagFlatModel
    fs_model = FileSystemFlatModel()
    
    tag_model = TagFlatModel()
    tag_model.load_tags() # Start Async Load
    
    grid_model = GalleryGridModel()
    
    # Journal Model
    from src.ui.models.journal_model import JournalViewModel
    journal_model = JournalViewModel()
    
    # Connect Journal Service -> View Model
    if sl.journal:
        # We use a lambda or direct method to push record to view model
        # Note: add_log must be thread-safe if called from thread. 
        # But here we are in Python land, GIL protects us mostly, but QAbstractListModel layoutChanged needs main thread.
        # Ideally we emit signal from service and connect explicitly.
        # For execution phase, direct call is acceptable if we are careful.
        # Better: use a Signal in Service? Service is pure python.
        # Let's use the callback we defined.
        sl.journal.set_ui_callback(journal_model.add_log)
    
    bridge = BackendBridge(importer, search_service, grid_model, journal_model)

    # 4. Setup Qt / Hybrid
    # Set Style to Fusion to support customization
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    
    # Check for PySide6-QtAds if we want to upgrade later, but for now standard docks
    
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    from src.ui.main_window import MainWindow
    window = MainWindow(bridge, fs_model, tag_model, grid_model, journal_model) # Window might need journal_model too if it uses it directly?
    # Window usually gets bridge. 
    # Let's check MainWindow signature.
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

