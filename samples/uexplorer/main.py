"""
UExplorer - Main Entry Point

Directory Opus-inspired file manager showcasing Foundation + UCoreFS.
"""
import sys
import asyncio
from pathlib import Path

# Add foundation to path
foundation_path = Path(__file__).parent.parent.parent / "templates" / "foundation"
sys.path.insert(0, str(foundation_path))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from qasync import QEventLoop
from loguru import logger

# Import Foundation and UCoreFS
from src.core.bootstrap import ApplicationBuilder
from src.core.locator import ServiceLocator
from src.ucorefs.core.fs_service import FSService
from src.ucorefs.discovery.service import DiscoveryService
from src.ucorefs.thumbnails.service import ThumbnailService
from src.ucorefs.vectors.service import VectorService
from src.ucorefs.ai.similarity_service import SimilarityService
from src.ucorefs.ai.llm_service import LLMService
from src.ucorefs.relations.service import RelationService
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.albums.manager import AlbumManager
from src.ucorefs.rules.engine import RulesEngine

# Import UExplorer UI
uex_ui_path = Path(__file__).parent / "src" / "ui"
sys.path.insert(0, str(uex_ui_path))
from main_window import MainWindow


async def async_init(qt_app, loop):
    """Async initialization of systems and window."""
    logger.info("-" * 100)
    logger.info("Step 4/7: ASYNC INITIALIZATION STARTING")
    logger.info("-" * 100)
    
    logger.info("  4.1: Building ApplicationBuilder...")
    config_path = Path(__file__).parent / "config.toml"
    logger.info(f"      Config path: {config_path}")
    logger.info(f"      Config exists: {config_path.exists()}")
    
    builder = (
        ApplicationBuilder("UExplorer", str(config_path))
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
        .add_system(RulesEngine)
    )
    
    logger.info("  4.2: Starting all systems...")
    locator = await builder.build()
    logger.info("  ✓ All 14 systems started successfully!")
    
    # Create main window (qasync handles event loop globally)
    logger.info("  4.3: Creating MainWindow...")
    window = MainWindow(locator)
    logger.info(f"  ✓ MainWindow created: {window}")
    logger.info(f"      Window title: {window.windowTitle()}")
    logger.info(f"      Window size: {window.size()}")
    
    # Keep window reference alive
    logger.info("  4.4: Storing window reference...")
    qt_app.main_window = window
    logger.info("  ✓ Window stored in qt_app.main_window")
    
    # Configure window
    logger.info("  4.5: Configuring window properties...")
    window.setAttribute(Qt.WA_DeleteOnClose, False)
    logger.info("  ✓ Set WA_DeleteOnClose = False (won't delete on close)")
    
    # Show window
    logger.info("  4.6: Showing window...")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint
    
    screen = QApplication.primaryScreen().geometry()
    x = (screen.width() - window.width()) // 2
    y = (screen.height() - window.height()) // 2
    window.move(max(0, x), max(0, y))
    
    logger.info(f"  ✓ Window centered at ({max(0, x)}, {max(0, y)})")
    
    window.show()
    logger.info("  ✓ Window.show() called")
    logger.info(f"      Is visible: {window.isVisible()}")
    logger.info(f"      Is active: {window.isActiveWindow()}")
    logger.info(f"      Window state:{window.windowState()}")
    
    # Bring window to front
    logger.info("  4.7: Bringing window to front...")
    window.raise_()
    window.activateWindow()
    logger.info(f"✓ Window visible: {window.isVisible()}")
    logger.info(f"✓ Window position: {window.pos()}")
    
    logger.info("-" * 100)
    logger.info("✓ ASYNC INITIALIZATION COMPLETE")
    logger.info("-" * 100)
    
    return window


def main():
    """Main entry point with qasync integration."""
    logger.info("=" * 100)
    logger.info("🚀 UEXPLORER STARTING")
    logger.info("=" * 100)
    
    exit_code = 0
    
    try:
        # Step 1: Create QApplication
        logger.info("Step 1/7: Creating QApplication...")
        qt_app = QApplication(sys.argv)
        qt_app.setApplicationName("UExplorer")
        qt_app.setOrganizationName("UCoreFS")
        logger.info(f"✓ QApplication created: {qt_app}")
        logger.info(f"  - Application name: {qt_app.applicationName()}")
        logger.info(f"  - Organization: {qt_app.organizationName()}")
        
        # Step 2: Configure quit behavior
        logger.info("Step 2/7: Configuring application quit behavior...")
        qt_app.setQuitOnLastWindowClosed(True)
        logger.info("✓ Set quitOnLastWindowClosed = True")
        logger.info("  - App will quit when main window closes")
        
        # Step 3: Create qasync event loop
        logger.info("Step 3/7: Creating qasync event loop...")
        loop = QEventLoop(qt_app)
        asyncio.set_event_loop(loop)
        logger.info(f"✓ Event loop created: {loop}")
        
        # Step 4-5: Initialize window
        logger.info("Step 5/7: Running async initialization...")
        window = loop.run_until_complete(async_init(qt_app, loop))
        logger.info(f"✓ Async initialization complete, window={window}")
        
        # Step 6: Verification
        logger.info("Step 6/7: Pre-event loop verification...")
        logger.info(f"  - Window visible: {window.isVisible()}")
        logger.info(f"  - Window size: {window.size()}")
        logger.info(f"  - Window position: {window.pos()}")
        logger.info(f"  - Has main_window ref: {hasattr(qt_app, 'main_window')}")
        logger.info(f"  - Loop running: {loop.is_running()}")
        logger.info(f"  - Loop closed: {loop.is_closed()}")
        
        # Step 7: Start event loop with qasync
        logger.info("=" * 100)
        logger.info("Step 7/7: STARTING QT EVENT LOOP")
        logger.info("=" * 100)
        logger.info("🎯 UExplorer is now running!")
        logger.info("🎯 Window should be visible and responsive")
        logger.info("🎯 Close the window or press Ctrl+C to exit")
        logger.info("=" * 100)
        
        # Use qasync's context manager
        logger.info("Calling loop.run_forever() with qasync...")
        logger.info("⚠️  CRITICAL: About to enter loop.run_forever()")
        logger.info(f"⚠️  Loop state: running={loop.is_running()}, closed={loop.is_closed()}")
        logger.info(f"⚠️  Qt app state: startingUp={qt_app.startingUp()}, closingDown={qt_app.closingDown()}")
        
        # Add aboutToQuit signal logging
        def on_about_to_quit():
            logger.critical("=" * 100)
            logger.critical("🚨 QApplication.aboutToQuit SIGNAL FIRED!")
            logger.critical("=" * 100)
            import traceback
            traceback.print_stack()
        
        qt_app.aboutToQuit.connect(on_about_to_quit)
        logger.info("✓ Connected aboutToQuit signal logger")
        
        # Schedule auto-load of roots after event loop starts (in qasync context)
        from PySide6.QtCore import QTimer
        def trigger_auto_load():
            logger.info("📂 Auto-loading library roots...")
            window.load_initial_roots()
        
        QTimer.singleShot(500, trigger_auto_load)
        logger.info("✓ Scheduled auto-load for 500ms after loop starts")
        
        with loop:
            logger.info("⚠️  ENTERING with loop: context")
            logger.info("⚠️  Calling loop.run_forever() NOW...")
            loop.run_forever()
            logger.info("⚠️  loop.run_forever() RETURNED!")
        
        logger.critical("=" * 100)
        logger.critical("🚨 EXITED qasync with loop: block")
        logger.critical(f"Event loop exited - Loop state: running={loop.is_running()}, closed={loop.is_closed()}")
        logger.critical("=" * 100)
        
    except KeyboardInterrupt:
        logger.info("=" * 100)
        logger.info("⚠️  KEYBOARD INTERRUPT - User requested shutdown")
        logger.info("=" * 100)
        exit_code = 0
        
    except Exception as e:
        logger.error("=" * 100)
        logger.error("❌ FATAL ERROR IN MAIN!")
        logger.error("=" * 100)
        logger.exception(f"Exception: {e}")
        exit_code = 1
    
    finally:
        # Cleanup
        logger.info("=" * 100)
        logger.info(f"EVENT LOOP EXITED - Exit code: {exit_code}")
        logger.info("=" * 100)
        
        logger.info("Reason: User closed window normally")
        
        # Graceful shutdown
        logger.info("Starting cleanup...")
        logger.info("  Stopping all systems...")
        # NOTE: Commenting out locator.stop_all() - it has async operations
        # that interfere with the event loop shutdown
        # try:
        #     locator = ServiceLocator.get_instance()
        #     locator.stop_all()
        #     logger.info("  ✓ All systems stopped")
        # except Exception as e:
        #     logger.error(f"Error during cleanup: {e}")
        logger.info("  ✓ Skipped stop_all (causes event loop issues)")
        
        logger.info("✓ Cleanup complete")
        
        logger.info("=" * 100)
        logger.info("👋 UEXPLORER SHUTDOWN COMPLETE")
        logger.info("=" * 100)
        
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
