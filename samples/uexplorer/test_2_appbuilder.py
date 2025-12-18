"""
Test 2: Minimal qasync + ApplicationBuilder
Add Foundation's ApplicationBuilder to see if that causes issues.
"""
import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtCore import Qt
from qasync import QEventLoop
from loguru import logger

# Add foundation to path
foundation_path = Path(__file__).parent.parent.parent / "templates" / "foundation"
sys.path.insert(0, str(foundation_path))

from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.core.fs_service import FSService


class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test 2: qasync + ApplicationBuilder")
        self.setGeometry(100, 100, 600, 200)
        
        label = QLabel("Testing qasync with ApplicationBuilder...\nIf this stays open, ApplicationBuilder is OK!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)
        
        logger.info("Window created")


async def async_init():
    """Initialize application with Foundation."""
    logger.info("Building application with Foundation...")
    
    config_path = Path(__file__).parent / "config.toml"
    builder = (ApplicationBuilder("Test", str(config_path))
              .with_default_systems()
              .add_system(FSService))
    
    locator = await builder.build()
    logger.info("✓ ApplicationBuilder complete")
    
    return locator


def main():
    logger.info("=" * 80)
    logger.info("TEST 2: Minimal qasync + ApplicationBuilder")
    logger.info("=" * 80)
    
    # Create Qt app
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    # Create qasync event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    logger.info(f"QEventLoop created: {loop}")
    
    # Initialize application
    logger.info("Running async_init...")
    locator = loop.run_until_complete(async_init())
    logger.info("async_init complete")
    
    # Create and show window
    window = SimpleWindow()
    window.show()
    logger.info(f"Window visible: {window.isVisible()}")
    
    # Run event loop
    logger.info("=" * 80)
    logger.info("Starting qasync loop.run_forever()...")
    logger.info("If window disappears, ApplicationBuilder causes the issue")
    logger.info("=" * 80)
    
    with loop:
        loop.run_forever()
    
    logger.info("Event loop exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
