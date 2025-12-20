"""
Draggable Tab Bar

Tab bar with drag & drop support for reordering and cross-container dragging.
"""
from typing import Optional
from PySide6.QtWidgets import QTabBar, QApplication
from PySide6.QtCore import Signal, Qt, QPoint, QMimeData, QRect
from PySide6.QtGui import QDrag, QPixmap, QPainter, QRegion, QMouseEvent
from loguru import logger


class DraggableTabBar(QTabBar):
    """
    Enhanced tab bar with drag & drop capabilities.
    
    Features:
    - Tab reordering within same bar
    - Drag tabs between containers
    - Visual drag preview
    - Drop indicator line
    - Window tearing (drag outside to float)
    
    Signals:
        tab_drag_started: Emitted when drag operation begins
        tab_dropped: Emitted when tab dropped within same bar
        tab_torn_off: Emitted when tab dragged outside bounds
    """
    
    tab_drag_started = Signal(str)  # document_id
    tab_dropped = Signal(str, int)  # document_id, target_index
    tab_torn_off = Signal(str, QPoint)  # document_id, global_pos
    drag_ended = Signal()  # Emitted when drag operation completes (success or cancel)
    
    def __init__(self, parent: Optional[QTabBar] = None):
        """
        Initialize draggable tab bar.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Enable drag & drop
        self.setAcceptDrops(True)
        self.setElideMode(Qt.ElideRight)
        self.setUsesScrollButtons(True)
        self.setMovable(False)  # We handle movement ourselves
        
        # Drag state
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_tab_index: int = -1
        self._drop_indicator_index: int = -1
        
        logger.debug("DraggableTabBar initialized")
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Record drag start position.
        
        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self._drag_tab_index = self.tabAt(event.pos())
            logger.debug(f"Mouse press on tab {self._drag_tab_index}")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Start drag if threshold exceeded.
        
        Args:
            event: Mouse move event
        """
        if not (event.buttons() & Qt.LeftButton):
            return
        
        if self._drag_tab_index < 0:
            return
        
        if not self._drag_start_pos:
            return
        
        # Check drag threshold
        distance = (event.pos() - self._drag_start_pos).manhattanLength()
        if distance < QApplication.startDragDistance():
            return
        
        # Start drag operation
        doc_id = self.tabData(self._drag_tab_index)
        if not doc_id:
            logger.warning(f"No document ID for tab {self._drag_tab_index}")
            return
        
        logger.info(f"Starting drag for document: {doc_id}")
        self.tab_drag_started.emit(doc_id)
        
        # Create drag
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-foundation-document", doc_id.encode())
        drag.setMimeData(mime)
        
        # Create preview pixmap
        pixmap = self._create_tab_preview(self._drag_tab_index)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        
        # Execute drag
        result = drag.exec(Qt.MoveAction)
        
        logger.debug(f"Drag completed with result: {result}")
        
        # Emit drag ended signal
        self.drag_ended.emit()
    
    def _create_tab_preview(self, index: int) -> QPixmap:
        """
        Create pixmap preview of tab for drag cursor.
        
        Args:
            index: Tab index to preview
            
        Returns:
            Pixmap of tab rendering
        """
        rect = self.tabRect(index)
        pixmap = QPixmap(rect.size())
        pixmap.fill(Qt.transparent)
        
        # Render tab to pixmap
        painter = QPainter(pixmap)
        self.render(painter, QPoint(), QRegion(rect))
        painter.end()
        
        logger.debug(f"Created tab preview: {rect.size()}")
        return pixmap
    
    def dragEnterEvent(self, event):
        """
        Accept drag if it's a document.
        
        Args:
            event: Drag enter event
        """
        if event.mimeData().hasFormat("application/x-foundation-document"):
            event.acceptProposedAction()
            logger.debug("Drag entered tab bar")
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """
        Show drop indicator at insertion point.
        
        Args:
            event: Drag move event
        """
        if not event.mimeData().hasFormat("application/x-foundation-document"):
            event.ignore()
            return
        
        # Calculate insertion index
        insert_index = self._get_drop_index(event.pos())
        
        if insert_index != self._drop_indicator_index:
            self._drop_indicator_index = insert_index
            self.update()  # Repaint to show indicator
        
        event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        """
        Hide drop indicator when drag leaves.
        
        Args:
            event: Drag leave event
        """
        self._drop_indicator_index = -1
        self.update()
        logger.debug("Drag left tab bar")
    
    def dropEvent(self, event):
        """
        Handle tab drop.
        
        Args:
            event: Drop event
        """
        if not event.mimeData().hasFormat("application/x-foundation-document"):
            event.ignore()
            return
        
        doc_id = event.mimeData().data("application/x-foundation-document").data().decode()
        insert_index = self._get_drop_index(event.pos())
        
        logger.info(f"Tab dropped: {doc_id} at index {insert_index}")
        self.tab_dropped.emit(doc_id, insert_index)
        
        self._drop_indicator_index = -1
        self.update()
        
        event.acceptProposedAction()
    
    def _get_drop_index(self, pos: QPoint) -> int:
        """
        Calculate insertion index from cursor position.
        
        Args:
            pos: Cursor position
            
        Returns:
            Index where tab should be inserted
        """
        for i in range(self.count()):
            rect = self.tabRect(i)
            if pos.x() < rect.center().x():
                return i
        
        # After last tab
        return self.count()
    
    def paintEvent(self, event):
        """
        Draw tabs and drop indicator.
        
        Args:
            event: Paint event
        """
        super().paintEvent(event)
        
        # Draw drop indicator if active
        if self._drop_indicator_index >= 0:
            painter = QPainter(self)
            painter.setPen(Qt.blue)
            
            # Get x position for indicator
            if self._drop_indicator_index < self.count():
                rect = self.tabRect(self._drop_indicator_index)
                x = rect.left()
            else:
                # After last tab
                if self.count() > 0:
                    rect = self.tabRect(self.count() - 1)
                    x = rect.right()
                else:
                    x = 0
            
            # Draw vertical line
            painter.drawLine(x, 0, x, self.height())
            painter.end()
