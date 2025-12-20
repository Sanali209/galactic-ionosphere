"""
Split management system for flexible document layouts.
Supports side-by-side, top-down, and nested splits.
"""
from typing import List, Optional, Dict, Any
from enum import Enum
from PySide6.QtWidgets import QWidget, QSplitter
from PySide6.QtCore import Qt
from loguru import logger
import uuid

class SplitOrientation(Enum):
    HORIZONTAL = Qt.Horizontal
    VERTICAL = Qt.Vertical

class SplitNode:
    """
    Represents a node in the split tree.
    Can be either a container (leaf) or a splitter (branch).
    """
    def __init__(self, node_id: str = None):
        self.id = node_id or str(uuid.uuid4())
        self.is_container = True  # True = holds docs, False = holds splits
        self.orientation: Optional[SplitOrientation] = None
        self.children: List['SplitNode'] = []
        self.sizes: List[int] = []  # For QSplitter size ratios
        self.container_widget = None  # Will hold SplitContainer instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "is_container": self.is_container,
            "orientation": self.orientation.name if self.orientation else None,
            "sizes": self.sizes,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitNode':
        """Deserialize from dict."""
        node = cls(data["id"])
        node.is_container = data["is_container"]
        if data["orientation"]:
            node.orientation = SplitOrientation[data["orientation"]]
        node.sizes = data["sizes"]
        node.children = [cls.from_dict(child) for child in data["children"]]
        return node

class SplitManager:
    """
    Manages the tree structure of split areas.
    Handles creation, removal, and navigation of splits.
    
    Enhanced with drag & drop support via DragCoordinator.
    """
    def __init__(self):
        # Start with single container
        self.root = SplitNode()
        self.active_node_id: Optional[str] = self.root.id
        
        # Initialize drag coordinator
        from .docking.drag_coordinator import DragCoordinator
        self.drag_coordinator = DragCoordinator(self)
        
        logger.info("SplitManager initialized with drag & drop support")
    
    def split_node(self, node_id: str, orientation: SplitOrientation) -> Optional[str]:
        """
        Split a container node into two parts.
        Returns ID of newly created container, or None if failed.
        """
        node = self._find_node(node_id, self.root)
        if not node or not node.is_container:
            logger.warning(f"Cannot split non-container node: {node_id}")
            return None
        
        # Convert this container into a splitter with 2 children
        node.is_container = False
        node.orientation = orientation
        
        # Save the old container widget's parent before we modify anything
        old_widget = node.container_widget
        old_parent = old_widget.parentWidget() if old_widget else None
        
        # Create two new containers
        child1 = SplitNode()
        child1.container_widget = old_widget  # Transfer existing widget
        
        child2 = SplitNode()
        # CRITICAL: Create a new container widget for child2!
        from .split_container import SplitContainer
        child2.container_widget = SplitContainer(child2.id)
        
        # Connect child2 to drag coordinator if we have one
        if self.drag_coordinator:
            self._connect_container(child2.container_widget)
        
        # CRITICAL: Create QSplitter to visually display both containers!
        splitter = QSplitter(orientation.value)  # Qt.Horizontal or Qt.Vertical
        
        # IMPORTANT: Set parent FIRST before adding widgets to avoid deletion
        if old_parent:
            splitter.setParent(old_parent)
        
        # Now add the container widgets to the splitter
        # This will reparent them but shouldn't delete their contents since splitter is already parented
        splitter.addWidget(child1.container_widget)
        splitter.addWidget(child2.container_widget)
        splitter.setSizes([1, 1])  # Equal split
        
        # Show the splitter
        splitter.show()
        
        # Replace old widget in parent's layout
        if old_parent and hasattr(old_parent, 'layout') and old_parent.layout():
            layout = old_parent.layout()
            # Find and replace old widget with splitter
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget() == old_widget:
                    layout.removeWidget(old_widget)
                    layout.insertWidget(i, splitter)
                    old_widget.setParent(None)  # Unparent the old widget before we modify it
                    break
        elif old_parent and isinstance(old_parent, QSplitter):
            # If parent is already a splitter, replace the widget
            index = old_parent.indexOf(old_widget)
            if index >= 0:
                old_parent.replaceWidget(index, splitter)
        elif old_parent and hasattr(old_parent, 'setCentralWidget'):
            # Parent is a QMainWindow - replace central widget to fill window
            old_parent.setCentralWidget(splitter)
            logger.debug("Replaced QMainWindow central widget with splitter")
        
        node.children = [child1, child2]
        node.sizes = [1, 1]  # Equal split
        node.container_widget = None
        
        logger.info(f"Split node {node_id} {orientation.name} into {child1.id}, {child2.id}")
        logger.debug(f"Created QSplitter with 2 containers")
        return child2.id
    
    def remove_split(self, node_id: str) -> bool:
        """
        Remove a container node and merge with sibling.
        Returns True if successful.
        """
        parent = self._find_parent(node_id, self.root)
        if not parent:
            logger.warning(f"Cannot remove root itself: {node_id}")
            return False
        
        # Find sibling
        siblings = [c for c in parent.children if c.id != node_id]
        if len(siblings) != 1:
            logger.error(f"Expected 1 sibling, found {len(siblings)}")
            return False
        
        sibling = siblings[0]
        
        # Replace parent with sibling in grandparent
        grandparent = self._find_parent(parent.id, self.root)
        if grandparent:
            idx = grandparent.children.index(parent)
            grandparent.children[idx] = sibling
        else:
            # Parent is root, replace root with sibling
            self.root = sibling
        
        logger.info(f"Removed split {node_id}, merged with {sibling.id}")
        return True
    
    def _connect_container(self, container):
        """Connect a container's drag signals to the coordinator."""
        container.document_drag_started.connect(
            lambda doc_id, container_id: self.drag_coordinator.start_drag(doc_id, container_id)
        )
        container.drag_ended.connect(self.drag_coordinator.end_drag)
        # Auto-remove splits when container becomes empty
        container.all_closed.connect(
            lambda nid=container.node_id: self._on_container_empty(nid)
        )
        logger.debug(f"Connected container {container.node_id} to drag coordinator")
    
    def _on_container_empty(self, node_id: str):
        """Handle when a container becomes empty - remove the split."""
        logger.info(f"Container {node_id} is empty, removing split")
        self.remove_split(node_id)
    
    def set_container_widget(self, node_id: str, widget):
        """Set container widget for a node and connect it."""
        node = self.get_node(node_id)
        if node:
            node.container_widget = widget
            self._connect_container(widget)
            logger.debug(f"Set container widget for node: {node_id}")
    
    def get_node(self, node_id: str) -> Optional[SplitNode]:
        """Get node by ID."""
        return self._find_node(node_id, self.root)
    
    def get_all_containers(self) -> List[SplitNode]:
        """Get all leaf container nodes."""
        containers = []
        self._collect_containers(self.root, containers)
        return containers
    
    def _find_node(self, node_id: str, current: SplitNode) -> Optional[SplitNode]:
        """Recursively find node by ID."""
        if current.id == node_id:
            return current
        for child in current.children:
            result = self._find_node(node_id, child)
            if result:
                return result
        return None
    
    def _find_parent(self, node_id: str, current: SplitNode) -> Optional[SplitNode]:
        """Find parent of node with given ID."""
        for child in current.children:
            if child.id == node_id:
                return current
            parent = self._find_parent(node_id, child)
            if parent:
                return parent
        return None
    
    def _collect_containers(self, node: SplitNode, containers: List[SplitNode]):
        """Recursively collect all container nodes."""
        if node.is_container:
            containers.append(node)
        else:
            for child in node.children:
                self._collect_containers(child, containers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire split tree."""
        return {
            "root": self.root.to_dict(),
            "active_node_id": self.active_node_id
        }
    
    def move_document(self, doc_id: str, from_container_id: str, to_container_id: str) -> bool:
        """
        Move document from one container to another.
        
        Args:
            doc_id: Document identifier
            from_container_id: Source container node ID
            to_container_id: Target container node ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        from_node = self.get_node(from_container_id)
        to_node = self.get_node(to_container_id)
        
        if not from_node or not to_node:
            logger.error(f"Invalid container IDs: from={from_container_id}, to={to_container_id}")
            return False
        
        if not from_node.container_widget or not to_node.container_widget:
            logger.error("Container widgets not available")
            return False
        
        # Get document from source
        doc = from_node.container_widget.remove_document(doc_id)
        if not doc:
            logger.warning(f"Document {doc_id} not found in {from_container_id}")
            return False
        
        # Add to target
        to_node.container_widget.add_document(doc)
        
        logger.info(f"Moved document {doc_id}: {from_container_id} -> {to_container_id}")
        return True
    
    def split_with_document(self, container_id: str, orientation, doc_id: str) -> Optional[str]:
        """
        Create new split and move document to it.
        
        Args:
            container_id: Container to split
            orientation: Qt.Horizontal or Qt.Vertical
            doc_id: Document to move to new container
            
        Returns:
            str: ID of new container, or None if failed
        """
        new_id = self.split_node(container_id, orientation)
        if new_id:
            self.move_document(doc_id, container_id, new_id)
        return new_id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitManager':
        """Deserialize split tree."""
        manager = cls()
        manager.root = SplitNode.from_dict(data["root"])
        manager.active_node_id = data.get("active_node_id")
        return manager
