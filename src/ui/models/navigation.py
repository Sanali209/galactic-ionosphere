from PySide6.QtCore import QObject, Signal, Slot, Property, QAbstractItemModel, QModelIndex, Qt, QDir
from PySide6.QtWidgets import QFileSystemModel
from src.core.database.models.tag import Tag
import asyncio
from loguru import logger

class FolderTreeModel(QFileSystemModel):
    """
    Wrapper around QFileSystemModel to provide specific filtering for the Gallery.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Drives)
        # Root path can be set later via property
        self.setRootPath("") 

class TagTreeModel(QAbstractItemModel):
    """
    Tree model for Tag hierarchy.
    Fetching data from MongoDB (async) and adapting to Qt (sync/slots).
    """
    
    # Custom Roles
    IdRole = Qt.UserRole + 1
    PathRole = Qt.UserRole + 2
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_items = [] # List of Tag objects (roots)
        self._tag_map = {} # id -> Tag object
        self._children_map = {} # id -> [Tag objects] (adjacency list)
        
        # We need a way to trigger async load
        # In a real app, strict separation via Bridge/Controller.
        # Here we might trigger load on init (fire-and-forget task).
        pass

    def load_tags(self):
        """Triggers reliable async loading."""
        asyncio.create_task(self._fetch_data())

    async def _fetch_data(self):
        self.beginResetModel()
        try:
            # Fetch all tags
            # Sort by path length or hierarchy?
            # Easiest: Fetch all, reconstruct tree in memory.
            all_tags = await Tag.find({})
            
            self._root_items = []
            self._tag_map = {}
            self._children_map = {}
            
            # 1. Map all
            for t in all_tags:
                self._tag_map[t.id] = t
                self._children_map[t.id] = []
                
            # 2. Build Hierarchy
            for t in all_tags:
                if t.parent_id:
                    if t.parent_id in self._children_map:
                        self._children_map[t.parent_id].append(t)
                    else:
                        # Parent missing? Treat as root?
                        self._root_items.append(t)
                else:
                    self._root_items.append(t)
                    
            logger.info(f"Loaded {len(all_tags)} tags into TreeModel.")
            
        except Exception as e:
            logger.error(f"Error loading tags: {e}")
            
        self.endResetModel()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            # Root level
            if row < len(self._root_items):
                return self.createIndex(row, column, self._root_items[row].id)
        else:
            # Child level
            parent_id = parent.internalId()
            children = self._children_map.get(parent_id, [])
            if row < len(children):
                return self.createIndex(row, column, children[row].id)
                
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_id = index.internalId()
        child = self._tag_map.get(child_id)
        
        if not child or not child.parent_id:
            return QModelIndex()
            
        parent_tag = self._tag_map.get(child.parent_id)
        if not parent_tag:
            return QModelIndex()
            
        # Find row of parent_tag in ITS parent
        if not parent_tag.parent_id:
            # It's a root item
            if parent_tag in self._root_items:
                row = self._root_items.index(parent_tag)
                return self.createIndex(row, 0, parent_tag.id)
        else:
            # It's a child of grandparent
            grandparent_children = self._children_map.get(parent_tag.parent_id, [])
            if parent_tag in grandparent_children:
                row = grandparent_children.index(parent_tag)
                return self.createIndex(row, 0, parent_tag.id)
                
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return len(self._root_items)
            
        parent_id = parent.internalId()
        return len(self._children_map.get(parent_id, []))

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        tag_id = index.internalId()
        tag = self._tag_map.get(tag_id)
        
        if not tag:
            return None
            
        if role == Qt.DisplayRole:
            return tag.name
        elif role == self.IdRole:
            return str(tag.id)
        elif role == self.PathRole:
            return tag.path
            
        return None
    
    def roleNames(self):
        return {
            Qt.DisplayRole: b"display",
            self.IdRole: b"tagId",
            self.PathRole: b"tagPath"
        }
