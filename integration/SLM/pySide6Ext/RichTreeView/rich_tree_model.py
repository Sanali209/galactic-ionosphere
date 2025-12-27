from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QObject, Signal, QThread, QMimeData
from PySide6.QtGui import QIcon
from SLM.pySide6Ext.RichTreeView.tree_node import TreeNode
from SLM.files_db.components.fs_tag import TagRecord

class ChildrenFetcher(QObject):
    """A worker to fetch children for a node in the background."""
    finished = Signal(QModelIndex, list)

    def __init__(self, parent_node, parent_index):
        super().__init__()
        self.parent_node = parent_node
        self.parent_index = parent_index

    def run(self):
        """Fetch children and emit the finished signal."""
        if isinstance(self.parent_node.data, TagRecord):
            child_tags = self.parent_node.data.child_tags()
            self.finished.emit(self.parent_index, child_tags)

class RichTreeModel(QAbstractItemModel):
    """A model for a tree view."""

    def __init__(self, headers, root_data=None, parent=None):
        super().__init__(parent)
        self._root_node = TreeNode(root_data)
        self.headers = headers
        self.threads = []
        self._is_fetching = set()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows under the given parent."""
        if not parent.isValid():
            parent_node = self._root_node
        else:
            parent_node = parent.internalPointer()
        return parent_node.child_count()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns."""
        return 1

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Return the data for the given index and role."""
        if not index.isValid():
            return None

        node = index.internalPointer()

        if role == Qt.ItemDataRole.DisplayRole:
            if hasattr(node.data, 'fullName'):
                return node.data.fullName
            return str(node.data)
        elif role == Qt.ItemDataRole.DecorationRole:
            # Example: return an icon
            pass

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        """Return the header data."""
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if section < len(self.headers):
                return self.headers[section]
        return None

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        """Return the index of the item in the model."""
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_node = self._root_node
        else:
            parent_node = parent.internalPointer()

        child_item = parent_node.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Return the parent of the model item."""
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        if not child_item or child_item == self._root_node:
            return QModelIndex()
            
        parent_item = child_item.parent()

        if parent_item is None or parent_item == self._root_node:
            return QModelIndex()

        # This check is defensive. If parent_item exists, it must have a row.
        if hasattr(parent_item, 'row'):
             return self.createIndex(parent_item.row(), 0, parent_item)
        
        return QModelIndex()

    def removeNode(self, node_to_remove):
        """Removes a node from the tree."""
        parent_node = node_to_remove.parent()
        if not parent_node:
            return False

        node_row = node_to_remove.row()
        
        # Get the parent index before removing the node
        if parent_node == self._root_node:
            parent_index = QModelIndex()
        else:
            parent_index = self.createIndex(parent_node.row(), 0, parent_node)

        self.beginRemoveRows(parent_index, node_row, node_row)
        parent_node.remove_child(node_to_remove)
        self.endRemoveRows()
        
        return True

    def get_root_node(self):
        return self._root_node

    def hasChildren(self, parent: QModelIndex = QModelIndex()) -> bool:
        """Check if a node has children."""
        if not parent.isValid():
            return True  # Root always has children

        node = parent.internalPointer()
        if isinstance(node.data, TagRecord):
            # Check if there are child tags without fetching them
            val = len(node.data.child_tags()) > 0
            return val
        return node.child_count() > 0

    def canFetchMore(self, parent: QModelIndex) -> bool:
        """Check if there is more data to fetch for a given parent."""
        if not parent.isValid():
            return False
        node = parent.internalPointer()
        return not node.children_loaded and parent not in self._is_fetching

    def fetchMore(self, parent: QModelIndex):
        """Fetch more data for a given parent."""
        if not parent.isValid() or parent in self._is_fetching:
            return

        parent_node = parent.internalPointer()
        if parent_node.children_loaded:
            return

        self._is_fetching.add(parent)
        thread = QThread()
        fetcher = ChildrenFetcher(parent_node, parent)
        fetcher.moveToThread(thread)

        thread.started.connect(fetcher.run)
        fetcher.finished.connect(self.on_children_fetched)
        fetcher.finished.connect(thread.quit)
        fetcher.finished.connect(fetcher.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.threads.append(thread)
        thread.start()

    def on_children_fetched(self, parent_index, children):
        """Slot to handle the fetched children."""
        parent_node = parent_index.internalPointer()
        
        if parent_node.children_loaded:
            self._is_fetching.discard(parent_index)
            return

        if children:
            self.beginInsertRows(parent_index, parent_node.child_count(), parent_node.child_count() + len(children) - 1)
            for child_data in children:
                child_node = TreeNode(child_data, parent_node)
                parent_node.append_child(child_node)
            self.endInsertRows()

        parent_node.children_loaded = True
        self._is_fetching.discard(parent_index)

    def flags(self, index):
        default_flags = super().flags(index)
        if index.isValid():
            return default_flags | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        else:
            return default_flags | Qt.ItemFlag.ItemIsDropEnabled

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return ["application/vnd.treeviewdragdrop.node"]

    def mimeData(self, indexes):
        mime_data = QMimeData()
        encoded_data = bytearray()
        for index in indexes:
            if index.isValid():
                node = index.internalPointer()
                # We'll use the memory address of the node to identify it.
                encoded_data.extend(str(id(node)).encode())
        mime_data.setData("application/vnd.treeviewdragdrop.node", encoded_data)
        return mime_data

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.DropAction.IgnoreAction:
            return True
        if not data.hasFormat("application/vnd.treeviewdragdrop.node"):
            return False
        if column > 0:
            return False

        parent_node = self.getNode(parent)

        encoded_data = data.data("application/vnd.treeviewdragdrop.node")
        node_address = int(bytes(encoded_data).decode())

        # Find the node from its memory address. This is a bit of a hack.
        # A better way would be to store nodes in a dictionary with a unique ID.
        dragged_node = self.find_node_by_address(self._root_node, node_address)

        if dragged_node:
            # Logic to move the node
            old_parent = dragged_node.parent()
            old_row = dragged_node.row()

            # Begin model manipulation
            self.beginMoveRows(self.createIndex(old_row, 0, old_parent), old_row, old_row, parent, row)
            
            # Reparent the node
            old_parent.remove_child(dragged_node)
            parent_node.insert_child(row, dragged_node)
            dragged_node.set_parent(parent_node)

            # End model manipulation
            self.endMoveRows()
            
            # Update the database
            if isinstance(dragged_node.data, TagRecord) and isinstance(parent_node.data, TagRecord):
                dragged_node.data.parent_tag = parent_node.data

            return True
        return False

    def getNode(self, index):
        if index.isValid():
            node = index.internalPointer()
            if node:
                return node
        return self._root_node

    def find_node_by_address(self, current_node, address):
        if id(current_node) == address:
            return current_node
        for i in range(current_node.child_count()):
            found = self.find_node_by_address(current_node.child(i), address)
            if found:
                return found
        return None
