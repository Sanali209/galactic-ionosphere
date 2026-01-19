"""
Tag Tree Model for UExplorer.

Implements a lazy-loading QAbstractItemModel for efficiently displaying
hierarchical tags from the database.
"""
from typing import List, Optional, Any, Dict
import asyncio
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal, Slot
from loguru import logger
from bson import ObjectId

from src.ucorefs.tags.models import Tag
from src.ucorefs.tags.manager import TagManager


class TreeNode:
    """Internal node structure for the model."""
    def __init__(self, tag: Optional[Tag] = None, parent: Optional['TreeNode'] = None):
        self.tag = tag
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.is_loaded = False  # True if children have been fetched
        self.is_fetching = False # True if fetch is in progress
        
        # If tag is None, it's the invisible root
        
    @property
    def id(self) -> Optional[ObjectId]:
        return self.tag._id if self.tag else None
        
    @property
    def name(self) -> str:
        return self.tag.name if self.tag else "Root"
        
    @property
    def row(self) -> int:
        if self.parent:
            return self.parent.children.index(self)
        return 0


class TagTreeModel(QAbstractItemModel):
    """
    Async Lazy-Loading Tree Model for Tags.
    """
    
    def __init__(self, tag_manager: TagManager, parent=None):
        super().__init__(parent)
        self.tag_manager = tag_manager
        self.root = TreeNode(None, None)
        
        # Initial load of root tags
        self.refresh_roots()
        
    def refresh_roots(self):
        """Fetch root tags."""
        self.beginResetModel()
        self.root = TreeNode(None, None)
        self.endResetModel()
        
        asyncio.ensure_future(self._fetch_children(self.root))

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
            
        parent_node = self.node_from_index(parent)
        if row < len(parent_node.children):
            return self.createIndex(row, column, parent_node.children[row])
            
        return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
            
        node = self.node_from_index(index)
        parent_node = node.parent
        
        if parent_node is None or parent_node == self.root:
            return QModelIndex()
            
        return self.createIndex(parent_node.row, 0, parent_node)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self.node_from_index(parent)
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
            
        node = self.node_from_index(index)
        
        if role == Qt.DisplayRole:
            if node.tag:
                count = node.tag.file_count or 0
                if count > 0:
                    return f"{node.tag.name} ({count})"
                return node.tag.name
                
        elif role == Qt.UserRole:
            # Return Tag ID as string
            return str(node.tag._id) if node.tag else None
            
        elif role == Qt.DecorationRole:
            # Could add icon here
            pass
            
        elif role == Qt.ForegroundRole:
            if node.tag and node.tag.color:
                try:
                    from PySide6.QtGui import QColor
                    return QColor(node.tag.color)
                except:
                    pass
                    
        return None

    def hasChildren(self, parent: QModelIndex = QModelIndex()) -> bool:
        node = self.node_from_index(parent)
        
        # If loaded, check actual children count
        if node.is_loaded:
            return len(node.children) > 0
            
        # If not loaded, always return True to show expansion indicator (lazy load)
        # Exception: Leaf nodes logic could be improved if we knew 'has_children' from DB
        # For now, assume folders might have children
        return True

    def canFetchMore(self, parent: QModelIndex) -> bool:
        node = self.node_from_index(parent)
        return not node.is_loaded and not node.is_fetching

    def fetchMore(self, parent: QModelIndex):
        node = self.node_from_index(parent)
        node.is_fetching = True
        logger.debug(f"Fetching children for {node.name}")
        asyncio.ensure_future(self._fetch_children(node))
        
    async def _fetch_children(self, parent_node: TreeNode):
        """Async fetcher."""
        try:
            parent_id = parent_node.id
            tags = await self.tag_manager.get_children(parent_id)
            
            # Switch to UI thread logic if needed? QAbstractItemModel methods must be on main thread
            # Since we are in async, we overlap with main loop.
            # But beginInsertRows emits signals, safe to do from async task in qasync loop?
            # Yes, if we are on the same thread/loop.
            
            if not tags:
                parent_node.is_loaded = True
                parent_node.is_fetching = False
                return

            # Update model
            # Find index if not root
            if parent_node == self.root:
                parent_index = QModelIndex()
            else:
                parent_index = self.createIndex(parent_node.row, 0, parent_node)
                
            self.beginInsertRows(parent_index, 0, len(tags) - 1)
            
            for tag in tags:
                new_node = TreeNode(tag, parent_node)
                parent_node.children.append(new_node)
                
            parent_node.is_loaded = True
            parent_node.is_fetching = False
            self.endInsertRows()
            
        except Exception as e:
            logger.error(f"Failed to fetch tags: {e}")
            parent_node.is_fetching = False

    def node_from_index(self, index: QModelIndex) -> TreeNode:
        if index.isValid():
            return index.internalPointer()
        return self.root

    def reload_node(self, tag_id: str):
        """
        Reload a specific node's children (refresh).
        TODO: Implement recursive refresh if needed.
        """
        # Finds node and clears loaded flag
        pass
