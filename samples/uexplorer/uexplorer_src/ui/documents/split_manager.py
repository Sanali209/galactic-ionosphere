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
    """
    def __init__(self):
        # Start with single container
        self.root = SplitNode()
        self.active_node_id: Optional[str] = self.root.id
        logger.info("SplitManager initialized with single container")
    
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
        
        # Create two new containers
        child1 = SplitNode()
        child1.container_widget = node.container_widget  # Transfer existing widget
        child2 = SplitNode()
        
        node.children = [child1, child2]
        node.sizes = [1, 1]  # Equal split
        node.container_widget = None
        
        logger.info(f"Split node {node_id} {orientation.name} into {child1.id}, {child2.id}")
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitManager':
        """Deserialize split tree."""
        manager = cls()
        manager.root = SplitNode.from_dict(data["root"])
        manager.active_node_id = data.get("active_node_id")
        return manager
