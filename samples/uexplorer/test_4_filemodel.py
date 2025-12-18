"""
Test 4: qasync + ApplicationBuilder + MainWindow + FileModel
Add FileModel with async root loading to see if that's the trigger.
"""
import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTreeView
from PySide6.QtCore import Qt
from qasync import QEventLoop
from loguru import logger

# Add foundation to path
foundation_path = Path(__file__).parent.parent.parent / "templates" / "foundation"
sys.path.insert(0, str(foundation_path))

from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.core.fs_service import FSService

# Import FileModel
models_path = Path(__file__).parent / "src" / "models"
sys.path.insert(0, str(models_path))
from file_model import FileModel


class TestMainWindow(QMainWindow):
    """MainWindow with FileModel but no dual-pane complexity."""
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        
        self.setWindowTitle("Test 4: qasync + MainWindow + FileModel")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central = QWidget()
        layout = QVBoxLayout()
        
        # Add label
        label = QLabel("Testing with FileModel...\n" +
                      "FileModel will load roots asynchronously\n" +
                      "If this exits immediately, FileModel is the culprit!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # Add tree view with FileModel
        logger.info("Creating QTreeView...")
        self.tree = QTreeView()
        layout.addWidget(self.tree)
        
        # Create FileModel
        logger.info("Creating FileModel (will trigger async root loading)...")
        self.model = FileModel(locator)
        self.tree.setModel(self.model)
        logger.info("✓ FileModel set on tree view")
        
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        logger.info("TestMainWindow created with FileModel")


async def async_init(qt_app):
    """Initialize application."""
    logger.info("Building application...")
    
    config_path = Path(__file__).parent / "config.toml"
    builder = (ApplicationBuilder("Test", str(config_path))
              .with_default_systems()
              .add_system(FSService))
    
    locator = await builder.build()
    logger.info("✓ ApplicationBuilder complete")
    
    # Create window with FileModel
    logger.info("Creating TestMainWindow with FileModel...")
    window = TestMainWindow(locator)
    window.show()
    logger.info(f"✓ Window visible: {window.isVisible()}")
    
    qt_app.main_window = window
    return window


def main():
    logger.info("=" * 80)
    logger.info("TEST 4: qasync + ApplicationBuilder + MainWindow + FileModel")
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
    logger.info("⚠️  CRITICAL: About to call loop.run_forever()")
    logger.info("⚠️  If app exits immediately, FileModel causes the issue!")
    logger.info("=" * 80)
    
    with loop:
        logger.info("⚠️  ENTERING loop.run_forever()...")
        loop.run_forever()
        logger.info("⚠️  loop.run_forever() RETURNED")
    
    logger.info("Event loop exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
