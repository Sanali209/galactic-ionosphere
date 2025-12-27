"""
Floating Window

Detached document window that can be re-docked.
Created when tab is dragged outside main window bounds.
"""
from typing import Optional
from PySide6.QtWidgets import QMainWindow, QWidget
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QMouseEvent, QMoveEvent
from loguru import logger


class FloatingWindow(QMainWindow):
    """
    Floating window for torn-off documents.
    
    Features:
    - Displays single document
    - Can be re-docked by dragging back
    - Maintains document state
    - Proper window lifecycle
    
    Signals:
        dock_requested: Emitted when user drags window back over main window
        closed: Emitted when window is closed
    """
    
    dock_requested = Signal(str, QPoint)  # document_id, global_position
    closed = Signal(str)  # document_id
    
    def __init__(self, document_view, parent: Optional[QWidget] = None):
        """
        Initialize floating window.
        
        Args:
            document_view: DocumentView to display
            parent: Parent widget (usually main window)
        """
        super().__init__(parent, Qt.Window)
        
        self.document = document_view
        self.setCentralWidget(document_view)
        self.setWindowTitle(document_view.title if hasattr(document_view, 'title') else "Document")
        
        # Track dragging state
        self._is_dragging = False
        self._drag_pos: Optional[QPoint] = None
        
        # Window configuration
        self.resize(800, 600)
        
        logger.info(f"FloatingWindow created for document: {getattr(document_view, 'id', 'unknown')}")
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Start tracking drag when titlebar clicked.
        
        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            logger.debug("Window drag started")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle window dragging and detect re-docking.
        
        Args:
            event: Mouse move event
        """
        if self._is_dragging and event.buttons() & Qt.LeftButton:
            if not self._drag_pos:
                return
            
            # Move window
            new_pos = event.globalPos() - self._drag_pos
            self.move(new_pos)
            
            # Check if over main window for re-docking
            main_window = self.parent()
            if main_window and isinstance(main_window, QWidget):
                main_rect = main_window.frameGeometry()
                if main_rect.contains(event.globalPos()):
                    doc_id = getattr(self.document, 'id', None)
                    if doc_id:
                        logger.debug(f"Window over main window, emitting dock_requested")
                        self.dock_requested.emit(doc_id, event.globalPos())
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Stop tracking drag.
        
        Args:
            event: Mouse release event
        """
        self._is_dragging = False
        logger.debug("Window drag ended")
        super().mouseReleaseEvent(event)
    
    def closeEvent(self, event):
        """
        Handle window close.
        
        Args:
            event: Close event
        """
        doc_id = getattr(self.document, 'id', None)
        if doc_id:
            logger.info(f"FloatingWindow closed for document: {doc_id}")
            self.closed.emit(doc_id)
        
        super().closeEvent(event)
    
    def get_document(self):
        """
        Get the document displayed in this window.
        
        Returns:
            DocumentView instance
        """
        return self.document
