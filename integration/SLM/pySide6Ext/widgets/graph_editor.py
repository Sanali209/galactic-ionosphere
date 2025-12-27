import math
from abc import abstractmethod
from math import sqrt

import networkx as nx
from PySide6.QtCore import QObject, QPointF, Qt, QRectF
from PySide6.QtGui import QContextMenuEvent, QPainter, QDragEnterEvent, QDropEvent, QAction, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, \
    QGraphicsSceneContextMenuEvent, QColorDialog, QGraphicsItem, QMenu, QLabel
from sympy import false
from tqdm import tqdm

from SLM.appGlue.core import ModeManager
from PySide6.QtCore import QSignalBlocker


# add procesing tags on tagg add and file add
# refactor save and load

class DataContextMixin(QObject):
    def __init__(self, data_context=None):
        super().__init__()
        self.data_context = data_context
        self.parent_scene = None

    def set_data_context(self, data_context):
        self.data_context = data_context

    @abstractmethod
    def on_add(self, parent_scene):
        self.parent_scene = parent_scene

    @abstractmethod
    def on_edit(self, parent_scene):
        pass

    @abstractmethod
    def on_delete(self, parent_scene):
        self.parent_scene = None

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        pass


class GraphScene(QGraphicsScene):
    """Custom QGraphicsScene to handle graph nodes and edges."""

    def __init__(self):
        super().__init__()
        self.mode_context = ModeManager()
        self.last_node = None
        self.nodes = []
        self.edges = []
        self.groups = []
        self.arrange_algorithm = "spring"  # Default layout algorithm
        self.all_arrange_algorithms = {
            "spring": self.arrange_graph_spring,
            "arf": self.arrange_graph_arf_layout,
            "kamada_kawai": self.arrange_kamada_kawai_layout,
            "shell": self.arrange_shell_layout,
            "spectral": self.arrange_spectral_layout,
            "spiral": self.arrange_graph_spiral_layout,
        }
        self.nodes_datacontext_dict = {}

    def set_arrange_algorithm(self, algorithm):
        """
        Set the layout algorithm for arranging the graph.

        Args:
            algorithm (str): The name of the layout algorithm.
        """
        if algorithm in self.all_arrange_algorithms:
            self.arrange_algorithm = algorithm
        else:
            raise ValueError(f"Unknown arrange algorithm: {algorithm}")

    def arrange_graph(self):
        """
        Arrange the graph nodes using the selected layout algorithm.
        """
        if self.arrange_algorithm in self.all_arrange_algorithms:
            self.all_arrange_algorithms[self.arrange_algorithm]()

    def arrange_graph_spring(self):
        """
        Arrange the graph nodes using the spring layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 512
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    if item is None and  edge.other_node(item) is None:
                        continue
                    graph.add_edge(item, edge.other_node(item))
        k = (1 / sqrt(len(self.nodes))) * 1.1
        pos = nx.spring_layout(graph, k=k)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))  # Scale positions
            for edge in node.edges:
                edge.update_position()

    def arrange_graph_arf_layout(self):
        """
        Arrange the graph nodes using the ARF layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 128
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    graph.add_edge(item, edge.other_node(item))
        pos = nx.arf_layout(graph)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))
            for edge in node.edges:
                edge.update_position()

    def arrange_kamada_kawai_layout(self):
        """
        Arrange the graph nodes using the Kamada-Kawai layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 256
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    graph.add_edge(item, edge.other_node(item))
        pos = nx.kamada_kawai_layout(graph)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))
            for edge in node.edges:
                edge.update_position()

    def arrange_shell_layout(self):
        """
        Arrange the graph nodes using the shell layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 512
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    graph.add_edge(item, edge.other_node(item))
        pos = nx.shell_layout(graph)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))
            for edge in node.edges:
                edge.update_position()

    def arrange_spectral_layout(self):
        """
        Arrange the graph nodes using the spectral layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 256
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    graph.add_edge(item, edge.other_node(item))
        pos = nx.spectral_layout(graph)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))
            for edge in node.edges:
                edge.update_position()

    def arrange_graph_spiral_layout(self):
        """
        Arrange the graph nodes using the spiral layout algorithm.

        Args:
            scene (GraphScene): The scene containing the graph nodes and edges.
        """
        if not self.nodes:
            return
        graph = nx.Graph()
        items_list = self.items()
        multipler = math.sqrt(len(self.nodes)) * 512
        for item in items_list:
            if isinstance(item, Node):
                graph.add_node(item, size=128)
                for edge in item.edges:
                    graph.add_edge(item, edge.other_node(item))
        pos = nx.spiral_layout(graph)
        for node, position in pos.items():
            node.setPos(QPointF(position[0] * multipler, position[1] * multipler))
            for edge in node.edges:
                edge.update_position()

    def mousePressEvent(self, event):
        """
        Respond to mouse clicks based on current_mode.
        """
        curent_mode = self.mode_context.current_mode
        if curent_mode:
            curent_mode.mousePressEvent(self, event)
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        curent_mode = self.mode_context.current_mode
        if curent_mode:
            curent_mode.keyPressEvent(self, event)

    def keyReleaseEvent(self, event):
        curent_mode = self.mode_context.current_mode
        if curent_mode:
            curent_mode.keyReleaseEvent(self, event)

    def suspend_view_updates(self, state=False):
        views = self.views()
        for view in views:
            view.setUpdatesEnabled(state)
            if state:
                view.update()

    def addItem(self, item):
        super().addItem(item)
        if isinstance(item, Node):
            # Check if the node already exists in the scene
            if item in self.nodes:
                return
            # Add the node to the list of nodes
            self.nodes.append(item)
            self.nodes_datacontext_dict[item.data_context] = item


        elif isinstance(item, Edge):
            self.edges.append(item)
            self.recalc_edges_position()

        elif isinstance(item, NodeGroup):
            self.groups.append(item)

        item.on_add(self)

    def addItems(self, items):
        self.suspend_view_updates()
        try:
            for item in tqdm(items):
                with QSignalBlocker(item):
                    self.addItem(item)
        finally:
            self.suspend_view_updates(True)

    def removeItem(self, item):
        super().removeItem(item)

        if isinstance(item, Node) and item in self.nodes:
            item.on_delete(self)
            # Remove the node from the list of nodes
            self.nodes.remove(item)
            if item.data_context in self.nodes_datacontext_dict:
                # Remove the node from the data context dictionary
                self.nodes_datacontext_dict.pop(item.data_context)

        elif isinstance(item, Edge) and item in self.edges:
            item.on_delete(self)
            #remove edges from nodes
            item.dispose()
            self.edges.remove(item)

        elif isinstance(item, NodeGroup) and item in self.groups:
            item.on_delete(self)
            self.groups.remove(item)

    def get_node_by_data_context(self, data_context):
        return self.nodes_datacontext_dict.get(data_context, None)

    def is_node_data_context_exists(self, data_context):
        return data_context in self.nodes_datacontext_dict

    def clear_scene(self):
        super().clear()
        self.last_node = None
        self.nodes.clear()
        self.edges.clear()
        self.groups.clear()

    def mouseMoveEvent(self, event):
        """
        Update any edges connected to the currently dragged node.
        """
        curent_mode = self.mode_context.current_mode
        if curent_mode:
            curent_mode.mouseMoveEvent(self, event)
        super().mouseMoveEvent(event)
        #for edge in self.edges:
        #edge.update_position()

    def mouseReleaseEvent(self, event):
        curent_mode = self.mode_context.current_mode
        if curent_mode:
            curent_mode.mouseReleaseEvent(self, event)
        super().mouseReleaseEvent(event)

    def recalc_edges_position(self):
        for edge in self.edges:
            edge.update_position()

    def set_mode(self, mode):
        self.current_mode = mode

    def get_selected_nodes(self,node_type=None):
        if node_type is not None:
            return [item for item in self.selectedItems() if isinstance(item, node_type)]
        else:
            # Return all selected nodes
            return [item for item in self.selectedItems() if isinstance(item, Node)]


    def get_selected_edges(self):
        return [item for item in self.selectedItems() if isinstance(item, Edge)]

    def get_selected_groups(self):
        return [item for item in self.selectedItems() if isinstance(item, NodeGroup)]

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if item:
            item.contextMenuEvent(event)

    def add_group(self, nodes):
        group = NodeGroup(nodes)
        self.addItem(group)
        self.groups.append(group)

    def delete_selected_items(self):
        for item in self.selectedItems():
            self.removeItem(item)

    def get_nodes_by_data_context_type(self, data_context_type):
        """
        Get all nodes with a specific data context type.

        Args:
            data_context_type (type): The type of data context to filter nodes by.

        Returns:
            list: A list of nodes with the specified data context type.
        """
        return [node for node in self.nodes if isinstance(node.data_context, data_context_type)]




class GraphView(QGraphicsView):
    """Custom QGraphicsView to render the GraphScene with antialiasing."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setSceneRect(scene.itemsBoundingRect())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._isMiddleMousePressed = False
        self._lastPos = None
        self.legend_label = QLabel(self)
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet("background-color: rgba(255, 255, 255, 0.8); border: 1px solid black;")



    def wheelEvent(self, event):
        #define zoom factor
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Determine zoom factor based on wheel direction
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # Apply the scaling
        self.scale(zoom_factor, zoom_factor)
        event.accept()

    def arrange_graph(self):
        self.scene().arrange_graph()
        groups = self.scene().groups
        for group in groups:
            group.update_bounding_rect()
        self.setSceneRect(self.scene().itemsBoundingRect())

    def FitInViewAll(self):
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def dragEnterEvent(self, event: QDragEnterEvent):
        curent_mode = self.scene().mode_context.current_mode
        if curent_mode:
            curent_mode.on_drag_enter(event)

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        curent_mode = self.scene().mode_context.current_mode
        if curent_mode:
            cur_scene = self.scene()
            curent_mode.on_drag_drop(event, cur_scene)

    def zoom_in(self):
        self.scale(1.2, 1.2)

    def zoom_out(self):
        self.scale(0.8, 0.8)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            # Only start panning on middle mouse button press
            self._isMiddleMousePressed = True
            self._lastPos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            # Process other buttons normally
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._isMiddleMousePressed:
            # Calculate the difference between the current and last mouse positions
            delta = event.pos() - self._lastPos
            self._lastPos = event.pos()
            # Adjust the scroll bars directly to simulate panning
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._isMiddleMousePressed = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class Node(QGraphicsEllipseItem, DataContextMixin):
    """Custom QGraphicsEllipseItem to represent a graph node."""

    def __init__(self, x, y, radius, data_context=None):

        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.set_data_context(data_context)

        self.setPos(x, y)
        self.radius = radius
        self.edges = []
        self.color = Qt.blue
        self.setBrush(self.color)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.image_path = None
        self.image = None
        self.parent_group = None

    def on_add(self, parent_scene):
        pass

    def on_edit(self, parent_scene):
        pass

    def on_delete(self, parent_scene):
        pass

    def set_image(self, image_path):
        from PySide6.QtGui import QPixmap
        self.image_path = image_path
        self.image = QPixmap(image_path)
        self.update()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.image:
            scaled_image = self.image.scaled(self.rect().width(), self.rect().height(), Qt.KeepAspectRatio)
            x_offset = (self.rect().width() - scaled_image.width()) / 2
            y_offset = (self.rect().height() - scaled_image.height()) / 2
            painter.drawPixmap(self.rect().x() + x_offset, self.rect().y() + y_offset, scaled_image)

    def add_edge(self, edge):
        """Add an edge to the node's list of edges."""
        self.edges.append(edge)

    def set_color(self):
        """Open a color dialog to set the node's color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.setBrush(color)

    def itemChange(self, change, value):
        """
        Handle item changes to update connected edges.

        Args:
            change (QGraphicsItem.GraphicsItemChange): The type of change.
            value (QVariant): The new value.

        Returns:
            QVariant: The processed value.
        """
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
            if self.parent_group is not None:
                self.parent_group.update_bounding_rect()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        pass


class Edge(QGraphicsLineItem, DataContextMixin):
    """Custom QGraphicsLineItem to represent an edge between two nodes."""

    def __init__(self, node1, node2, data_context=None):
        """
        Initialize an edge between two nodes.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.
        """
        super().__init__()
        self.set_data_context(data_context)
        self.color = Qt.black
        self.edge_size = 6
        self.setPen(QPen(self.color, self.edge_size))
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.setFlag(QGraphicsLineItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsLineItem.ItemIsMovable, False)
        self.setFlag(QGraphicsLineItem.ItemSendsGeometryChanges, True)
        self.setZValue(-1)  # Draw edges behind nodes
        self.update_position()
        self.relation_record = None

    def set_color(self, color):
        self.color = color
        self.setPen(QPen(color, self.edge_size))

    def other_node(self, node):
        """
        Get the other node connected by this edge.

        Args:
            node (Node): One of the nodes connected by this edge.

        Returns:
            Node: The other node connected by this edge.
        """
        return self.node2 if self.node1 == node else self.node1

    def update_position(self):
        """
        Update the position of the edge using each node's center.
        """
        x1 = self.node1.scenePos().x()
        y1 = self.node1.scenePos().y()
        x2 = self.node2.scenePos().x()
        y2 = self.node2.scenePos().y()
        self.setLine(x1, y1, x2, y2)

    def itemChange(self, change, value):
        """
        Update edge line when the edge itself is moved.
        """
        if change == QGraphicsItem.ItemPositionChange:
            self.update_position()
        return super().itemChange(change, value)

    def on_add(self, parent_scene):
        pass

    def on_edit(self, parent_scene):
        pass

    def on_delete(self, parent_scene):
        pass

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        menu = QMenu()
        delete_action = QAction("Delete Edge", menu)
        delete_action.triggered.connect(lambda: self.delete())
        menu.addAction(delete_action)

        menu.exec(event.screenPos())

    def mousePressEvent(self, event):
        pass

    def dispose(self):
        self.node1.edges.remove(self)
        self.node2.edges.remove(self)
        self.node1 = None
        self.node2 = None

class NodeGroup(QGraphicsRectItem, DataContextMixin):

    def __init__(self, nodes, data_context=None):
        super().__init__()
        self.set_data_context(data_context)
        self.nodes = nodes
        self.update_bounding_rect()
        self.setBrush(Qt.green)
        self.setOpacity(0.3)
        self.setFlags(
            QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        for node in nodes:
            node.parent_group = self
            node.setParentItem(self)

        #set z order -2
        self.setZValue(-2)

    def update_bounding_rect(self):
        """Update the bounding rect to fit all child nodes."""
        if not self.nodes:
            self.prepareGeometryChange()  # Notify Qt about geometry changes
            self.setRect(QRectF())  # Reset bounding rect if no nodes
            return

        # Convert scene bounding rects of child nodes to local coordinates
        rect = self.mapFromScene(self.nodes[0].sceneBoundingRect()).boundingRect()

        for node in self.nodes[1:]:
            node_rect = self.mapFromScene(node.sceneBoundingRect()).boundingRect()
            rect = rect.united(node_rect)  # Expand bounding rect to fit all children

        self.prepareGeometryChange()  # Notify Qt that geometry is about to change
        self.setRect(rect)  # Update the bounding rect

    def setPos(self, pos):
        """Override setPos to update child nodes."""
        delta = pos - self.pos()
        super().setPos(pos)
        for node in self.nodes:
            node.setPos(node.pos() + delta)

    def itemChange(self, change, value):
        """Обновление позиций дочерних узлов при перемещении группы."""
        if change == QGraphicsItem.ItemPositionChange:
            delta = value - self.pos()
            for node in self.nodes:
                node.setPos(node.pos() + delta)
        return super().itemChange(change, value)

    def on_add(self, parent_scene):
        pass

    def on_edit(self, parent_scene):
        pass

    def on_delete(self, parent_scene):
        pass

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        """Create context menu for nodes, edges, and groups."""
        menu = QMenu()
        delete_action = QAction("Delete Group", menu)
        delete_action.triggered.connect(lambda: self.scene().removeItem(self))
        menu.addAction(delete_action)
        menu.exec(event.screenPos())
