# -*- coding: utf-8 -*-
"""
PinItem - Visual representation of a node pin.

Pins are the connection points on nodes where wires attach.
They handle mouse interaction for creating connections.
"""
from typing import Optional, TYPE_CHECKING, List
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsItem
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QBrush, QPen, QFont

if TYPE_CHECKING:
    from .node_item import NodeItem
    from .connection_item import ConnectionItem
    from ..core.pins import BasePin


class PinItem(QGraphicsEllipseItem):
    """
    Visual representation of a node pin.
    
    Pins are small circles on the sides of nodes.
    - Input pins: Left side
    - Output pins: Right side
    
    Attributes:
        pin_data: The underlying BasePin
        node_item: Parent NodeItem
        is_input: True for input pins, False for output
        connections: List of connected ConnectionItems
    """
    
    PIN_RADIUS = 6
    PIN_HOVER_RADIUS = 8
    
    def __init__(
        self, 
        pin_data: 'BasePin',
        node_item: 'NodeItem',
        is_input: bool = True,
        parent: Optional[QGraphicsItem] = None
    ):
        """
        Create a pin item.
        
        Args:
            pin_data: The underlying pin
            node_item: Parent node item
            is_input: True for input, False for output
            parent: Parent graphics item
        """
        super().__init__(parent)
        
        self.pin_data = pin_data
        self.node_item = node_item
        self.is_input = is_input
        self.connections: List['ConnectionItem'] = []
        
        # Setup appearance
        self._setup_appearance()
        
        # Make interactive
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(10)  # Above node body
    
    def _setup_appearance(self):
        """Configure visual appearance based on pin type."""
        # Get color from pin type
        color_hex = self.pin_data.pin_type.color
        self._color = QColor(color_hex)
        self._hover_color = self._color.lighter(130)
        
        # Set size
        r = self.PIN_RADIUS
        self.setRect(-r, -r, r * 2, r * 2)
        
        # Fill and border
        self._update_brush(False)
    
    def _update_brush(self, hovered: bool):
        """Update brush based on hover state."""
        color = self._hover_color if hovered else self._color
        
        if self.is_connected():
            # Filled when connected
            self.setBrush(QBrush(color))
        else:
            # Hollow when not connected
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        self.setPen(QPen(color, 2))
    
    def is_connected(self) -> bool:
        """Check if pin has any connections."""
        return len(self.connections) > 0
    
    def can_connect_to(self, other: 'PinItem') -> bool:
        """Check if connection to another pin is valid."""
        if not other or other == self:
            return False
        
        # Same node not allowed
        if other.node_item == self.node_item:
            return False
        
        # Use pin data validation
        return self.pin_data.can_connect_to(other.pin_data)
    
    def get_scene_center(self) -> QPointF:
        """Get pin center position in scene coordinates."""
        return self.scenePos()
    
    def add_connection(self, connection: 'ConnectionItem'):
        """Register a connection to this pin."""
        if connection not in self.connections:
            self.connections.append(connection)
            self._update_brush(False)
    
    def remove_connection(self, connection: 'ConnectionItem'):
        """Unregister a connection from this pin."""
        if connection in self.connections:
            self.connections.remove(connection)
            self._update_brush(False)
    
    def update_connections(self):
        """Update all connection positions."""
        for conn in self.connections:
            conn.update_path()
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def hoverEnterEvent(self, event):
        """Handle mouse hover enter."""
        self._update_brush(True)
        # Expand slightly
        r = self.PIN_HOVER_RADIUS
        self.setRect(-r, -r, r * 2, r * 2)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave."""
        self._update_brush(False)
        # Return to normal size
        r = self.PIN_RADIUS
        self.setRect(-r, -r, r * 2, r * 2)
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press - start connection drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Tell parent widget to start connection creation
            view = self.scene().views()[0] if self.scene() and self.scene().views() else None
            if view and hasattr(view, 'start_connection_drag'):
                view.start_connection_drag(self)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def __repr__(self) -> str:
        direction = "in" if self.is_input else "out"
        return f"<PinItem {self.pin_data.name} ({direction})>"
