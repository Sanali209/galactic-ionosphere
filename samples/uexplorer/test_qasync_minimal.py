"""
Minimal qasync test - just show a window with qasync event loop.
"""
import sys
import asyncio
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtCore import Qt
from qasync import QEventLoop
from loguru import logger


class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("qasync Test Window")
        self.setGeometry(100, 100, 400, 200)
        
        label = QLabel("If you see this and window stays open, qasync works!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)
        
        logger.info("Window created")
    
    def closeEvent(self, event):
        logger.info("Window closing")
        event.accept()


def main():
    logger.info("=" * 80)
    logger.info("Starting minimal qasync test")
    logger.info("=" * 80)
    
    # Create Qt app
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    logger.info("QApplication created")
    
    # Create qasync event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    logger.info(f"QEventLoop created: {loop}")
    
    # Create and show window
    window = SimpleWindow()
    window.show()
    logger.info(f"Window visible: {window.isVisible()}")
    
    # Run event loop
    logger.info("Starting event loop with qasync...")
    logger.info("If window disappears immediately, qasync has a problem")
    logger.info("=" * 80)
    
    with loop:
        loop.run_forever()
    
    logger.info("Event loop exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
