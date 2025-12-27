"""
Minimal Docking Demo - Test basic QtAds functionality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PySide6.QtCore import Qt
import PySide6QtAds as QtAds
from loguru import logger


def main():
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    app = QApplication(sys.argv)
    
    logger.info("Creating main window...")
    window = QMainWindow()
    window.setWindowTitle("Minimal Docking Test")
    window.resize(800, 600)
    
    logger.info("Creating DockManager...")
    dock_manager = QtAds.CDockManager(window)
    window.setCentralWidget(dock_manager)
    
    logger.info("Creating dock widget 1...")
    dock1 = QtAds.CDockWidget("Editor")
    dock1.setWidget(QTextEdit())
    dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, dock1)
    
    logger.info("Creating dock widget 2...")
    dock2 = QtAds.CDockWidget("Properties")
    dock2.setWidget(QLabel("Properties Panel"))
    dock_manager.addDockWidget(QtAds.RightDockWidgetArea, dock2)
    
    logger.info("Showing window...")
    window.show()
    
    logger.info("Starting app...")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
