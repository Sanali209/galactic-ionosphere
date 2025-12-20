# -*- coding: utf-8 -*-
"""
ConnectionItem - Visual representation of a connection between pins.

Draws a bezier curve between two pins with color based on pin type.
"""
from typing import Optional, TYPE_CHECKING
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsItem
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPen, QPainterPath, QPainter

if TYPE_CHECKING:
    from .pin_item import PinItem
    from ..core.connection import NodeConnection


class ConnectionItem(QGraphicsPathItem):
    """
    Visual bezier curve connection between two pins.
    
    The curve is colored based on the pin type and has
    a smooth bezier shape for visual clarity.
    
    Attributes:
        connection_data: The underlying NodeConnection (optional during drag)
        source_pin: Source PinItem (output)
        target_pin: Target PinItem (input)
    """
    
    DEFAULT_COLOR = QColor(200, 200, 200)
    SELECTED_COLOR = QColor(255, 180, 0)
    HOVER_COLOR = QColor(255, 255, 255)
    LINE_WIDTH = 2.5
    
    def __init__(
        self, 
        source_pin: 'PinItem' = None,
        target_pin: 'PinItem' = None,
        connection_data: 'NodeConnection' = None,
        parent: Optional[QGraphicsItem] = None
    ):
        """
        Create a connection item.
        
        Args:
            source_pin: Source pin (output)
            target_pin: Target pin (input), None during drag
            connection_data: Underlying connection data
            parent: Parent graphics item
        """
        super().__init__(parent)
        
        self.connection_data = connection_data
        self.source_pin = source_pin
        self.target_pin = target_pin
        self._temp_end_point: Optional[QPointF] = None
        self._hovered = False
        
        # Setup appearance
        self._setup_appearance()
        
        # Make interactive
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-1)  # Behind nodes
        
        # Register with pins
        if source_pin:
            source_pin.add_connection(self)
        if target_pin:
            target_pin.add_connection(self)
        
        # Initial path
        self.update_path()
    
    def _setup_appearance(self):
        """Configure visual appearance."""
        # Get color from pin type
        if self.source_pin:
            color = QColor(self.source_pin.pin_data.pin_type.color)
        else:
            color = self.DEFAULT_COLOR
        
        self._color = color
        self._pen = QPen(color, self.LINE_WIDTH)
        self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(self._pen)
    
    def update_path(self, temp_end: QPointF = None):
        """
        Update the bezier path.
        
        Args:
            temp_end: Temporary end point during drag
        """
        if temp_end:
            self._temp_end_point = temp_end
        
        # Get start and end points
        if self.source_pin:
            start = self.source_pin.get_scene_center()
        else:
            return  # Can't draw without source
        
        if self.target_pin:
            end = self.target_pin.get_scene_center()
        elif self._temp_end_point:
            end = self._temp_end_point
        else:
            return  # Can't draw without end
        
        # Create bezier path
        path = self._create_bezier_path(start, end)
        self.setPath(path)
    
    def _create_bezier_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        """
        Create a smooth bezier curve between two points.
        
        Args:
            start: Start point (output pin)
            end: End point (input pin)
            
        Returns:
            QPainterPath with bezier curve
        """
        path = QPainterPath()
        path.moveTo(start)
        
        # Calculate control points for smooth curve
        dx = abs(end.x() - start.x())
        dy = abs(end.y() - start.y())
        
        # Horizontal distance for control points
        ctrl_dist = max(dx * 0.5, min(dx, 100), 50)
        
        ctrl1 = QPointF(start.x() + ctrl_dist, start.y())
        ctrl2 = QPointF(end.x() - ctrl_dist, end.y())
        
        path.cubicTo(ctrl1, ctrl2, end)
        
        return path
    
    def set_complete(self, target_pin: 'PinItem', connection_data: 'NodeConnection'):
        """
        Complete the connection by setting target pin.
        
        Called when drag-connect is finished successfully.
        
        Args:
            target_pin: Target pin item
            connection_data: The created connection
        """
        self.target_pin = target_pin
        self.connection_data = connection_data
        self._temp_end_point = None
        
        # Register with target pin
        target_pin.add_connection(self)
        
        self.update_path()
    
    def disconnect(self):
        """Remove this connection from pins."""
        if self.source_pin:
            self.source_pin.remove_connection(self)
        if self.target_pin:
            self.target_pin.remove_connection(self)
        
        # Remove from scene
        if self.scene():
            self.scene().removeItem(self)
    
    # =========================================================================
    # Visual Feedback
    # =========================================================================
    
    def paint(self, painter: QPainter, option, widget=None):
        """Custom paint for selection/hover feedback."""
        # Update pen based on state
        if self.isSelected():
            pen = QPen(self.SELECTED_COLOR, self.LINE_WIDTH + 1)
        elif self._hovered:
            pen = QPen(self.HOVER_COLOR, self.LINE_WIDTH + 0.5)
        else:
            pen = self._pen
        
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        
        super().paint(painter, option, widget)
    
    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)
    
    def __repr__(self) -> str:
        src = self.source_pin.pin_data.name if self.source_pin else "?"
        tgt = self.target_pin.pin_data.name if self.target_pin else "?"
        return f"<ConnectionItem {src} -> {tgt}>"
