# -*- coding: utf-8 -*-
"""
NodeGraphWidget - Main canvas for editing node graphs.

Provides a QGraphicsView-based canvas with:
- Pan and zoom navigation
- Node creation/deletion
- Connection drag-and-drop
- Selection and multi-select
- Clipboard operations
"""
from typing import Dict, Optional, List, TYPE_CHECKING
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QMenu, QApplication
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QColor, QBrush, QPen, QPainter, QWheelEvent, 
    QMouseEvent, QKeyEvent, QContextMenuEvent
)
from loguru import logger

if TYPE_CHECKING:
    from ..core.graph import NodeGraph
    from ..core.base_node import BaseNode
    from ..core.registry import NodeRegistry

from .node_item import NodeItem
from .pin_item import PinItem
from .connection_item import ConnectionItem


class NodeGraphWidget(QGraphicsView):
    """
    Main canvas widget for editing node graphs.
    
    Features:
    - Pan: Middle-mouse drag or Space+drag
    - Zoom: Mouse wheel
    - Selection: Click, Ctrl+Click, box select
    - Delete: Delete key
    - Copy/Paste: Ctrl+C/V
    
    Signals:
        node_selected: Emitted when node selection changes
        connection_created: Emitted when connection is made
        graph_changed: Emitted when graph is modified
    """
    
    # Signals
    node_selected = Signal(object)  # BaseNode or None
    connection_created = Signal(object)  # NodeConnection
    graph_changed = Signal()
    
    # Constants
    GRID_SIZE = 20
    ZOOM_MIN = 0.1
    ZOOM_MAX = 4.0
    ZOOM_STEP = 1.15
    
    def __init__(self, parent=None):
        """Create the node graph widget."""
        super().__init__(parent)
        
        self._graph: Optional['NodeGraph'] = None
        self._registry: Optional['NodeRegistry'] = None
        self._node_items: Dict[str, NodeItem] = {}
        self._connection_items: Dict[str, ConnectionItem] = {}
        
        # Interaction state
        self._panning = False
        self._pan_start = QPointF()
        self._space_pressed = False
        self._dragging_connection: Optional[ConnectionItem] = None
        self._drag_start_pin: Optional[PinItem] = None
        
        # Setup scene
        self._setup_scene()
        
        # Setup view
        self._setup_view()
    
    def _setup_scene(self):
        """Initialize the graphics scene."""
        scene = QGraphicsScene(self)
        scene.setSceneRect(-5000, -5000, 10000, 10000)
        scene.setBackgroundBrush(QBrush(QColor(30, 30, 32)))
        self.setScene(scene)
    
    def _setup_view(self):
        """Configure view settings."""
        # Rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Viewport
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Drag mode for selection
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # Focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Accept drops from palette
        self.setAcceptDrops(True)
    
    # =========================================================================
    # Graph Management
    # =========================================================================
    
    def set_graph(self, graph: 'NodeGraph', registry: 'NodeRegistry' = None):
        """
        Set the graph to display.
        
        Args:
            graph: NodeGraph to display
            registry: NodeRegistry for node creation
        """
        self._clear_display()
        self._graph = graph
        self._registry = registry
        self._rebuild_display()
    
    def get_graph(self) -> Optional['NodeGraph']:
        """Get the current graph."""
        return self._graph
    
    def _clear_display(self):
        """Clear all visual items."""
        self.scene().clear()
        self._node_items.clear()
        self._connection_items.clear()
    
    def _rebuild_display(self):
        """Rebuild display from graph data."""
        if not self._graph:
            return
        
        # Create node items
        for node in self._graph.nodes.values():
            self._add_node_item(node)
        
        # Create connection items
        for conn in self._graph.connections.values():
            self._add_connection_item(conn)
    
    def _add_node_item(self, node: 'BaseNode') -> NodeItem:
        """Create and add a NodeItem for a node."""
        item = NodeItem(node)
        self.scene().addItem(item)
        self._node_items[node.node_id] = item
        return item
    
    def _add_connection_item(self, conn) -> Optional[ConnectionItem]:
        """Create and add a ConnectionItem for a connection."""
        # Find pin items
        source_node_item = self._node_items.get(conn.source_pin.node.node_id)
        target_node_item = self._node_items.get(conn.target_pin.node.node_id)
        
        if not source_node_item or not target_node_item:
            return None
        
        source_pin_item = source_node_item.output_pins.get(conn.source_pin.name)
        target_pin_item = target_node_item.input_pins.get(conn.target_pin.name)
        
        if not source_pin_item or not target_pin_item:
            return None
        
        item = ConnectionItem(source_pin_item, target_pin_item, conn)
        self.scene().addItem(item)
        self._connection_items[conn.connection_id] = item
        return item
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def add_node(self, node_type: str, position: QPointF = None) -> Optional[NodeItem]:
        """
        Add a new node to the graph.
        
        Args:
            node_type: Type of node to create
            position: Position in scene coordinates
            
        Returns:
            Created NodeItem or None
        """
        if not self._graph or not self._registry:
            return None
        
        node = self._registry.create_node(node_type)
        if not node:
            logger.warning(f"Unknown node type: {node_type}")
            return None
        
        # Set position
        if position:
            node.position = (position.x(), position.y())
        
        # Add to graph
        self._graph.add_node(node)
        
        # Create visual
        item = self._add_node_item(node)
        
        self.graph_changed.emit()
        return item
    
    def delete_selected(self):
        """Delete selected nodes and connections."""
        if not self._graph:
            return
        
        # Get selected items
        selected = self.scene().selectedItems()
        
        # Delete connections first
        for item in selected:
            if isinstance(item, ConnectionItem):
                if item.connection_data:
                    self._graph.disconnect(item.connection_data.connection_id)
                    del self._connection_items[item.connection_data.connection_id]
                item.disconnect()
        
        # Delete nodes (this also removes their connections)
        for item in selected:
            if isinstance(item, NodeItem):
                node_id = item.node_data.node_id
                self._graph.remove_node(node_id)
                del self._node_items[node_id]
                self.scene().removeItem(item)
        
        self.graph_changed.emit()
    
    # =========================================================================
    # Connection Creation
    # =========================================================================
    
    def start_connection_drag(self, pin_item: PinItem):
        """
        Start dragging a new connection from a pin.
        
        Args:
            pin_item: The pin to start from
        """
        self._drag_start_pin = pin_item
        
        # Create temporary connection
        self._dragging_connection = ConnectionItem(
            source_pin=pin_item if not pin_item.is_input else None,
            target_pin=pin_item if pin_item.is_input else None,
        )
        self.scene().addItem(self._dragging_connection)
    
    def _update_connection_drag(self, scene_pos: QPointF):
        """Update the dragging connection position."""
        if self._dragging_connection:
            self._dragging_connection.update_path(scene_pos)
    
    def _finish_connection_drag(self, target_pin: Optional[PinItem] = None):
        """
        Finish connection drag.
        
        Args:
            target_pin: Target pin if successful, None if cancelled
        """
        if not self._dragging_connection or not self._drag_start_pin:
            return
        
        success = False
        
        if target_pin and target_pin != self._drag_start_pin:
            # Check if connection is valid
            if self._drag_start_pin.can_connect_to(target_pin):
                # Determine source/target based on direction
                if self._drag_start_pin.is_input:
                    source_pin = target_pin
                    target_pin_final = self._drag_start_pin
                else:
                    source_pin = self._drag_start_pin
                    target_pin_final = target_pin
                
                # Create connection in graph
                if self._graph:
                    try:
                        conn = self._graph.connect(
                            source_pin.node_item.node_data.node_id,
                            source_pin.pin_data.name,
                            target_pin_final.node_item.node_data.node_id,
                            target_pin_final.pin_data.name
                        )
                        
                        # Update visual
                        self._dragging_connection.disconnect()
                        self.scene().removeItem(self._dragging_connection)
                        
                        # Create proper connection item
                        conn_item = self._add_connection_item(conn)
                        
                        self.connection_created.emit(conn)
                        self.graph_changed.emit()
                        success = True
                        
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to create connection: {e}")
        
        if not success:
            # Cancel - remove temporary connection
            self._dragging_connection.disconnect()
            self.scene().removeItem(self._dragging_connection)
        
        self._dragging_connection = None
        self._drag_start_pin = None
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw grid background."""
        super().drawBackground(painter, rect)
        
        # Draw grid
        pen = QPen(QColor(50, 50, 52), 1)
        painter.setPen(pen)
        
        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)
        
        lines = []
        for x in range(left, int(rect.right()), self.GRID_SIZE):
            lines.append((QPointF(x, rect.top()), QPointF(x, rect.bottom())))
        for y in range(top, int(rect.bottom()), self.GRID_SIZE):
            lines.append((QPointF(rect.left(), y), QPointF(rect.right(), y)))
        
        for p1, p2 in lines:
            painter.drawLine(p1, p2)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel."""
        if event.angleDelta().y() > 0:
            factor = self.ZOOM_STEP
        else:
            factor = 1 / self.ZOOM_STEP
        
        # Check zoom limits
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        
        if self.ZOOM_MIN <= new_scale <= self.ZOOM_MAX:
            self.scale(factor, factor)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        # Middle mouse or Space+Left for panning
        if event.button() == Qt.MouseButton.MiddleButton or \
           (self._space_pressed and event.button() == Qt.MouseButton.LeftButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        # Panning
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        
        # Connection dragging
        if self._dragging_connection:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_connection_drag(scene_pos)
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        # End panning
        if self._panning and (event.button() == Qt.MouseButton.MiddleButton or
                              event.button() == Qt.MouseButton.LeftButton):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        
        # End connection dragging
        if self._dragging_connection:
            scene_pos = self.mapToScene(event.position().toPoint())
            target_item = self.scene().itemAt(scene_pos, self.transform())
            
            target_pin = None
            if isinstance(target_item, PinItem):
                target_pin = target_item
            
            self._finish_connection_drag(target_pin)
            event.accept()
            return
        
        # Handle selection
        super().mouseReleaseEvent(event)
        
        # Emit selection change
        selected = self.scene().selectedItems()
        if selected and isinstance(selected[0], NodeItem):
            self.node_selected.emit(selected[0].node_data)
        else:
            self.node_selected.emit(None)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press."""
        modifiers = event.modifiers()
        key = event.key()
        
        if key == Qt.Key.Key_Space:
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key == Qt.Key.Key_Delete:
            self.delete_selected()
        elif key == Qt.Key.Key_F:
            self.fit_in_view()
        elif key == Qt.Key.Key_D and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.duplicate_selected()
        elif key == Qt.Key.Key_C and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.copy_selected()
        elif key == Qt.Key.Key_V and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.paste()
        elif key == Qt.Key.Key_X and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.cut_selected()
        elif key == Qt.Key.Key_A and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.select_all()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release."""
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)
    
    def dragEnterEvent(self, event):
        """Accept drag from palette."""
        if event.mimeData().hasFormat("application/x-nodetype"):
            event.acceptProposedAction()
        elif event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """Accept drag move."""
        if event.mimeData().hasFormat("application/x-nodetype") or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop from palette."""
        node_type = None
        
        if event.mimeData().hasFormat("application/x-nodetype"):
            node_type = bytes(event.mimeData().data("application/x-nodetype")).decode()
        elif event.mimeData().hasText():
            node_type = event.mimeData().text()
        
        if node_type and self._registry:
            scene_pos = self.mapToScene(event.pos())
            self.add_node(node_type, scene_pos)
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def contextMenuEvent(self, event: QContextMenuEvent):
        """Handle right-click context menu."""
        menu = QMenu(self)
        
        scene_pos = self.mapToScene(event.pos())
        item = self.scene().itemAt(scene_pos, self.transform())
        
        if isinstance(item, NodeItem) or isinstance(item, ConnectionItem):
            # Item context menu
            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(self.delete_selected)
            
            if isinstance(item, NodeItem):
                menu.addSeparator()
                duplicate_action = menu.addAction("Duplicate (Ctrl+D)")
                duplicate_action.triggered.connect(self.duplicate_selected)
                
                copy_action = menu.addAction("Copy (Ctrl+C)")
                copy_action.triggered.connect(self.copy_selected)
                
                cut_action = menu.addAction("Cut (Ctrl+X)")
                cut_action.triggered.connect(self.cut_selected)
        else:
            # Empty space context menu - show node types
            paste_action = menu.addAction("Paste (Ctrl+V)")
            paste_action.triggered.connect(self.paste)
            paste_action.setEnabled(bool(self._clipboard))
            
            menu.addSeparator()
            
            if self._registry:
                categories = self._registry.get_categories()
                for category, nodes in sorted(categories.items()):
                    submenu = menu.addMenu(category)
                    for node_cls in nodes:
                        action = submenu.addAction(
                            node_cls.metadata.display_name or node_cls.node_type
                        )
                        # Capture node_type for lambda
                        node_type = node_cls.node_type
                        action.triggered.connect(
                            lambda checked, nt=node_type, pos=scene_pos: 
                            self.add_node(nt, pos)
                        )
        
        menu.exec(event.globalPos())
    
    def fit_in_view(self):
        """Fit all nodes in view."""
        if self._node_items:
            # Calculate bounding rect of all nodes
            rects = [item.sceneBoundingRect() for item in self._node_items.values()]
            combined = rects[0]
            for r in rects[1:]:
                combined = combined.united(r)
            
            # Add padding
            combined.adjust(-50, -50, 50, 50)
            
            self.fitInView(combined, Qt.AspectRatioMode.KeepAspectRatio)
    
    # =========================================================================
    # Clipboard Operations
    # =========================================================================
    
    _clipboard: List[dict] = []  # Class-level clipboard
    
    def copy_selected(self):
        """Copy selected nodes to clipboard."""
        selected = [item for item in self.scene().selectedItems() 
                    if isinstance(item, NodeItem)]
        
        if not selected:
            return
        
        self._clipboard.clear()
        
        for item in selected:
            node_data = item.node_data.to_dict()
            self._clipboard.append(node_data)
        
        logger.debug(f"Copied {len(self._clipboard)} nodes")
    
    def cut_selected(self):
        """Cut selected nodes (copy + delete)."""
        self.copy_selected()
        self.delete_selected()
    
    def paste(self):
        """Paste nodes from clipboard."""
        if not self._clipboard or not self._graph or not self._registry:
            return
        
        # Calculate offset from original positions
        offset = 50
        
        # Clear current selection
        self.scene().clearSelection()
        
        new_items = []
        
        for node_data in self._clipboard:
            node_type = node_data.get("node_type")
            node = self._registry.create_node(node_type)
            
            if node:
                # Offset position
                pos = node_data.get("position", [0, 0])
                node.position = (pos[0] + offset, pos[1] + offset)
                
                # Restore pin values
                for pin_name, value in node_data.get("pin_values", {}).items():
                    pin = node.get_input_pin(pin_name)
                    if pin:
                        pin.default_value = value
                
                self._graph.add_node(node)
                item = self._add_node_item(node)
                item.setSelected(True)
                new_items.append(item)
        
        if new_items:
            self.graph_changed.emit()
            logger.debug(f"Pasted {len(new_items)} nodes")
    
    def duplicate_selected(self):
        """Duplicate selected nodes."""
        self.copy_selected()
        self.paste()
    
    def select_all(self):
        """Select all nodes."""
        for item in self._node_items.values():
            item.setSelected(True)
    
    def get_node_item(self, node_id: str) -> Optional[NodeItem]:
        """Get NodeItem by node ID."""
        return self._node_items.get(node_id)
    
    def highlight_node(self, node_id: str, scroll_to: bool = True):
        """
        Highlight a node (for error display).
        
        Args:
            node_id: ID of node to highlight
            scroll_to: Whether to scroll view to the node
        """
        item = self._node_items.get(node_id)
        if item:
            self.scene().clearSelection()
            item.setSelected(True)
            
            if scroll_to:
                self.centerOn(item)
