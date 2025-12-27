# -*- coding: utf-8 -*-
"""
NodeItem - Visual representation of a node.

Displays a node with:
- Header bar with title and color
- Pin items for inputs (left) and outputs (right)
- Selection and drag support
- Error state visualization
"""
from typing import Dict, Optional, TYPE_CHECKING
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsDropShadowEffect, QStyleOptionGraphicsItem, QWidget
)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QColor, QBrush, QPen, QFont, QPainter, 
    QPainterPath, QLinearGradient
)

if TYPE_CHECKING:
    from ..core.base_node import BaseNode

from .pin_item import PinItem


class NodeItem(QGraphicsItem):
    """
    Visual representation of a node in the graph.
    
    Features:
    - Colored header with node title
    - Input pins on left, output pins on right
    - Drag to move
    - Selection highlight
    - Error state (red border)
    
    Attributes:
        node_data: The underlying BaseNode
        pin_items: Dict of pin name -> PinItem
    """
    
    # Visual constants
    HEADER_HEIGHT = 24
    PIN_SPACING = 20
    PIN_OFFSET = 12
    MIN_WIDTH = 120
    CORNER_RADIUS = 6
    BORDER_WIDTH = 2
    
    def __init__(self, node_data: 'BaseNode', parent: Optional[QGraphicsItem] = None):
        """
        Create a node item.
        
        Args:
            node_data: The underlying BaseNode
            parent: Parent graphics item
        """
        super().__init__(parent)
        
        self.node_data = node_data
        self.input_pins: Dict[str, PinItem] = {}
        self.output_pins: Dict[str, PinItem] = {}
        
        # Calculate dimensions
        self._calculate_size()
        
        # Create visual elements
        self._create_pins()
        
        # Setup interactivity
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # Set initial position from node data
        self.setPos(node_data.position[0], node_data.position[1])
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(3, 3)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
        
        # Set tooltip
        self._update_tooltip()
    
    def _calculate_size(self):
        """Calculate node size based on pins."""
        num_inputs = len(self.node_data.input_pins)
        num_outputs = len(self.node_data.output_pins)
        max_pins = max(num_inputs, num_outputs, 1)
        
        # Calculate title width
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        title = self.node_data.metadata.display_name or self.node_data.node_type
        
        # Estimate width needed for text
        self._width = max(self.MIN_WIDTH, len(title) * 8 + 40)
        self._height = self.HEADER_HEIGHT + max_pins * self.PIN_SPACING + self.PIN_OFFSET
    
    def _create_pins(self):
        """Create PinItem for each pin."""
        # Input pins on left
        input_pins = list(self.node_data.input_pins.values())
        for i, pin in enumerate(input_pins):
            y = self.HEADER_HEIGHT + self.PIN_OFFSET + i * self.PIN_SPACING
            pin_item = PinItem(pin, self, is_input=True, parent=self)
            pin_item.setPos(0, y)
            self.input_pins[pin.name] = pin_item
        
        # Output pins on right
        output_pins = list(self.node_data.output_pins.values())
        for i, pin in enumerate(output_pins):
            y = self.HEADER_HEIGHT + self.PIN_OFFSET + i * self.PIN_SPACING
            pin_item = PinItem(pin, self, is_input=False, parent=self)
            pin_item.setPos(self._width, y)
            self.output_pins[pin.name] = pin_item
    
    def boundingRect(self) -> QRectF:
        """Return bounding rectangle."""
        return QRectF(0, 0, self._width, self._height)
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        """Paint the node."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.boundingRect()
        
        # Determine colors
        header_color = QColor(self.node_data.metadata.color)
        body_color = QColor(40, 40, 42)
        
        # Selection/error border
        if self.node_data.has_error:
            border_color = QColor(255, 60, 60)
            border_width = 3
        elif self.isSelected():
            border_color = QColor(255, 180, 0)
            border_width = 2
        else:
            border_color = QColor(60, 60, 62)
            border_width = 1
        
        # Draw body with rounded corners
        path = QPainterPath()
        path.addRoundedRect(rect, self.CORNER_RADIUS, self.CORNER_RADIUS)
        
        # Body gradient
        body_gradient = QLinearGradient(0, 0, 0, rect.height())
        body_gradient.setColorAt(0, body_color.lighter(110))
        body_gradient.setColorAt(1, body_color)
        
        painter.setBrush(QBrush(body_gradient))
        painter.setPen(QPen(border_color, border_width))
        painter.drawPath(path)
        
        # Draw header
        header_rect = QRectF(0, 0, rect.width(), self.HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, self.CORNER_RADIUS, self.CORNER_RADIUS)
        
        # Clip bottom corners of header
        clip_rect = QRectF(0, self.CORNER_RADIUS, rect.width(), self.HEADER_HEIGHT - self.CORNER_RADIUS)
        header_path.addRect(clip_rect)
        
        header_gradient = QLinearGradient(0, 0, 0, self.HEADER_HEIGHT)
        header_gradient.setColorAt(0, header_color.lighter(120))
        header_gradient.setColorAt(1, header_color)
        
        painter.setBrush(QBrush(header_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setClipRect(QRectF(0, 0, rect.width(), self.HEADER_HEIGHT))
        painter.drawPath(header_path)
        painter.setClipping(False)
        
        # Draw title
        title = self.node_data.metadata.display_name or self.node_data.node_type
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        
        title_rect = QRectF(8, 0, rect.width() - 16, self.HEADER_HEIGHT)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, title)
        
        # Draw pin labels
        self._draw_pin_labels(painter)
    
    def _draw_pin_labels(self, painter: QPainter):
        """Draw labels next to pins."""
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 180))
        
        label_offset = 14
        
        # Input labels (right-aligned, left side)
        for name, pin_item in self.input_pins.items():
            y = pin_item.pos().y()
            rect = QRectF(label_offset, y - 8, self._width / 2 - label_offset - 5, 16)
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
        
        # Output labels (left-aligned, right side)
        for name, pin_item in self.output_pins.items():
            y = pin_item.pos().y()
            rect = QRectF(self._width / 2 + 5, y - 8, self._width / 2 - label_offset - 5, 16)
            painter.drawText(rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
    
    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update node data position
            pos = self.pos()
            self.node_data.position = (pos.x(), pos.y())
            
            # Update connection paths
            for pin in list(self.input_pins.values()) + list(self.output_pins.values()):
                pin.update_connections()
        
        return super().itemChange(change, value)
    
    def get_pin_item(self, pin_name: str) -> Optional[PinItem]:
        """Get PinItem by name."""
        return self.input_pins.get(pin_name) or self.output_pins.get(pin_name)
    
    def _update_tooltip(self):
        """Update tooltip based on node state."""
        lines = []
        
        # Node info
        title = self.node_data.metadata.display_name or self.node_data.node_type
        lines.append(f"<b>{title}</b>")
        
        if self.node_data.metadata.description:
            lines.append(f"<i>{self.node_data.metadata.description}</i>")
        
        lines.append(f"ID: {self.node_data.node_id[:8]}...")
        
        # Error info
        if self.node_data.has_error:
            lines.append("")
            lines.append(f"<font color='#FF4444'><b>ERROR:</b> {self.node_data.error_message}</font>")
        
        self.setToolTip("<br>".join(lines))
    
    def update_error_state(self):
        """Refresh display based on error state."""
        self._update_tooltip()
        self.update()
    
    def __repr__(self) -> str:
        return f"<NodeItem {self.node_data.node_type}({self.node_data.node_id[:8]})>"
