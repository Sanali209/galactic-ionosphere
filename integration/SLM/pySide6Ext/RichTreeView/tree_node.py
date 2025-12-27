from __future__ import annotations
from typing import Any, List

class TreeNode:
    """A node in the tree structure."""
    def __init__(self, data: Any, parent: TreeNode | None = None):
        self._data = data
        self._parent = parent
        self._children: List[TreeNode] = []
        self.children_loaded = False

    def append_child(self, child: TreeNode):
        """Add a child to this node."""
        self._children.append(child)

    def insert_child(self, row: int, child: TreeNode):
        """Insert a child at a specific row."""
        self._children.insert(row, child)

    def remove_child(self, child: TreeNode):
        """Remove a child from this node."""
        self._children.remove(child)

    def child(self, row: int) -> TreeNode | None:
        """Return the child at a specific row."""
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def child_count(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def parent(self) -> TreeNode | None:
        """Return the parent of this node."""
        return self._parent

    def set_parent(self, parent: TreeNode | None):
        """Set the parent of this node."""
        self._parent = parent

    def row(self) -> int:
        """Return the row of this node within its parent's children."""
        if self._parent:
            return self._parent._children.index(self)
        return 0

    @property
    def data(self) -> Any:
        """Return the data associated with this node."""
        return self._data
