"""
Test 3: Minimal qasync + ApplicationBuilder + MainWindow (empty)
Add a simplified MainWindow to see if MainWindow itself causes the exit.
"""
import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from qasync import QEventLoop
from loguru import logger

# Add foundation to path
foundation_path = Path(__file__).parent.parent.parent / "templates" / "foundation"
sys.path.insert(0, str(foundation_path))

from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.core.fs_service import FSService


class SimpleMainWindow(QMainWindow):
    """Simplified MainWindow - just a label, no file panes."""
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        
        self.setWindowTitle("Test 3: qasync + ApplicationBuilder + MainWindow")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget with label
        central = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("Testing qasync with simplified MainWindow...\n" +
                      "No FilePanes, no FileModel\n" +
                      "If this stays open, MainWindow structure is OK!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        logger.info("SimpleMainWindow created")


async def async_init(qt_app):
    """Initialize application with Foundation."""
    logger.info("Building application...")
    
    config_path = Path(__file__).parent / "config.toml"
    builder = (ApplicationBuilder("Test", str(config_path))
              .with_default_systems()
              .add_system(FSService))
    
    locator = await builder.build()
    logger.info("✓ ApplicationBuilder complete")
    
    # Create window
    logger.info("Creating SimpleMainWindow...")
    window = SimpleMainWindow(locator)
    window.show()
    logger.info(f"✓ Window visible: {window.isVisible()}")
    
    qt_app.main_window = window
    return window


def main():
    logger.info("=" * 80)
    logger.info("TEST 3: qasync + ApplicationBuilder + MainWindow")
    logger.info("=" * 80)
    
    # Create Qt app
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    # Create qasync event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    logger.info(f"QEventLoop created")
    
    # Initialize application
    logger.info("Running async_init...")
    window = loop.run_until_complete(async_init(app))
    logger.info("async_init complete")
    
    # Run event loop
    logger.info("=" * 80)
    logger.info("Starting qasync loop.run_forever()...")
    logger.info("If window disappears, MainWindow causes the issue")
    logger.info("=" * 80)
    
    with loop:
        loop.run_forever()
    
    logger.info("Event loop exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
