"""
Detection Graphics Items.
Custom QGraphicsItems for displaying and editing detection bounding boxes.
"""
from typing import Optional, Callable
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem, QGraphicsRectItem, QGraphicsSceneHoverEvent, QGraphicsSceneMouseEvent
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QCursor

class ResizeHandle(QGraphicsRectItem):
    """
    Handle for resizing a detection box.
    """
    def __init__(self, cursor: Qt.CursorShape, parent: QGraphicsItem, on_move: Callable):
        super().__init__(-4, -4, 8, 8, parent)
        self.setCursor(cursor)
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(QColor("black"), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        # We handle movement manually via parent's mouse events usually, 
        # or we make this movable and notify parent.
        # Better: Parent handles hover/click, or this handle notifies parent.
        # Implementation choice: Make handle selectable/movable?
        # Simpler: This is just visual, parent intercepts mouse near corners?
        # Or standard: ItemIsMovable = True, on position change notify parent.
        # Let's try explicit mouse handling.
        self.setAcceptHoverEvents(True)
        self._on_move = on_move
        self._dragging = False

    def mousePressEvent(self, event):
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            # Notify parent of the new position in parent coords
            new_pos = self.mapToParent(event.pos())
            self._on_move(self, new_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        event.accept()


class DetectionRectItem(QGraphicsObject):
    """
    Editable detection bounding box.
    """
    
    # Signals
    geometry_changed = Signal()
    modified = Signal()  # End of interaction
    
    def __init__(self, x, y, w, h, label: str = "", score: float = 0.0, color: str = "#00ff00", det_id: str = "", parent=None):
        super().__init__(parent)
        self.det_id = det_id
        self._rect = QRectF(x, y, w, h)
        self._label = label
        self._score = score
        self._color = QColor(color)
        self._is_editable = False
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # Border
        self._pen = QPen(self._color, 2)
        self._pen.setCosmetic(True) # Width stays constant on zoom
        
        # Handles
        self._handles = []
        self._init_handles()
        self._update_handles()
        
    def _init_handles(self):
        """Create resize handles."""
        # Top-Left, Top-Right, Bot-Right, Bot-Left
        cursors = [Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor, 
                   Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor]
        
        # Callbacks for each handle
        def move_tl(h, pos): self.setRect(QRectF(pos.x(), pos.y(), self._rect.right() - pos.x(), self._rect.bottom() - pos.y()))
        def move_tr(h, pos): self.setRect(QRectF(self._rect.left(), pos.y(), pos.x() - self._rect.left(), self._rect.bottom() - pos.y()))
        def move_br(h, pos): self.setRect(QRectF(self._rect.left(), self._rect.top(), pos.x() - self._rect.left(), pos.y() - self._rect.top()))
        def move_bl(h, pos): self.setRect(QRectF(pos.x(), self._rect.top(), self._rect.right() - pos.x(), pos.y() - self._rect.top()))
        
        callbacks = [move_tl, move_tr, move_br, move_bl]
        
        for i in range(4):
            h = ResizeHandle(cursors[i], self, callbacks[i])
            h.hide()
            self._handles.append(h)
            
    def setRect(self, rect: QRectF):
        """Update rectangle (normalized logic if needed)."""
        # Ensure positive width/height
        self._rect = rect.normalized()
        self.prepareGeometryChange()
        self._update_handles()
        self.geometry_changed.emit()
        self.update()

    def rect(self) -> QRectF:
        return self._rect

    def boundingRect(self) -> QRectF:
        # Include pen width and handle size
        margin = 10 
        return self._rect.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget):
        if self.isSelected():
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
        else:
            painter.setPen(self._pen)
            
        painter.drawRect(self._rect)
        
        # Draw Label
        painter.setPen(Qt.GlobalColor.white)
        # Background for text
        text = f"{self._label} {self._score:.2f}"
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        
        text_rect = QRectF(self._rect.left(), self._rect.top() - text_h, text_w + 4, text_h)
        painter.fillRect(text_rect, self._color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            is_selected =  bool(value.value) if hasattr(value, 'value') else bool(value)
            for h in self._handles:
                h.setVisible(is_selected and self._is_editable)
        
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.isSelected():
            # If moved, we might want to announce 'modified' on release based on mouseRelease
            pass

        return super().itemChange(change, value)
    
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable:
            self.modified.emit()

    def _update_handles(self):
        """Position handles at corners."""
        r = self._rect
        corners = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
        for i, h in enumerate(self._handles):
            h.setPos(corners[i])
            
    def set_editable(self, editable: bool):
        self._is_editable = editable
        # Show handles if selected
        if self.isSelected():
            for h in self._handles:
                h.setVisible(editable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editable)
