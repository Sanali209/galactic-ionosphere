"""
Test 5: FileModel WITHOUT auto-loading roots
Remove the QTimer.singleShot from FileModel.__init__ to see if that's the trigger.
"""
import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTreeView, QPushButton
from PySide6.QtCore import Qt
from qasync import QEventLoop, asyncSlot
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
    """MainWindow with FileModel - manually load roots."""
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        
        self.setWindowTitle("Test 5: FileModel WITHOUT auto-load")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central = QWidget()
        layout = QVBoxLayout()
        
        # Add label
        label = QLabel("Test 5: FileModel without QTimer auto-load\n" +
                      "Click button to manually trigger root loading\n" +
                      "If this stays open, QTimer was the culprit!")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # Add button to manually load
        btn = QPushButton("Load Roots Manually")
        btn.clicked.connect(self.load_roots_manually)
        layout.addWidget(btn)
        
        # Add tree view with FileModel
        self.tree = QTreeView()
        layout.addWidget(self.tree)
        
        # Create FileModel (comment out auto-load)
        logger.info("Creating FileModel (NO auto-load)...")
        self.model = FileModel(locator)
        
        # HACK: Clear the QTimer that FileModel sets up
        # This tests if QTimer is the issue
        logger.info("⚠️  Preventing FileModel auto-load by not calling anything")
        
        self.tree.setModel(self.model)
        logger.info("✓ FileModel set on tree view (roots NOT loaded)")
        
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        logger.info("TestMainWindow created")
    
    def load_roots_manually(self):
        """Manually load roots when button clicked."""
        logger.info("Button clicked - loading roots via refresh_roots()")
        self.model.refresh_roots()


async def async_init(qt_app):
    """Initialize application."""
    logger.info("Building application...")
    
    config_path = Path(__file__).parent / "config.toml"
    builder = (ApplicationBuilder("Test", str(config_path))
              .with_default_systems()
              .add_system(FSService))
    
    locator = await builder.build()
    logger.info("✓ ApplicationBuilder complete")
    
    # Create window
    logger.info("Creating TestMainWindow...")
    window = TestMainWindow(locator)
    window.show()
    logger.info(f"✓ Window visible: {window.isVisible()}")
    
    qt_app.main_window = window
    return window


def main():
    logger.info("=" * 80)
    logger.info("TEST 5: FileModel WITHOUT QTimer auto-load")
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
    logger.info("⚠️  If window stays open, QTimer was the culprit!")
    logger.info("⚠️  Click 'Load Roots' button to test manual loading")
    logger.info("=" * 80)
    
    with loop:
        logger.info("⚠️  Calling loop.run_forever()...")
        loop.run_forever()
        logger.info("⚠️  loop.run_forever() RETURNED")
    
    logger.info("Event loop exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
