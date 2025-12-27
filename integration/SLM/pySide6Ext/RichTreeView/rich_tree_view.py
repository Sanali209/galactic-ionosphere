from PySide6.QtWidgets import QTreeView, QMenu
from PySide6.QtCore import Qt, QSortFilterProxyModel, Signal, QModelIndex
from SLM.pySide6Ext.RichTreeView.rich_tree_model import RichTreeModel
from SLM.pySide6Ext.RichTreeView.rich_item_delegate import RichItemDelegate
from SLM.pySide6Ext.RichTreeView.tree_node import TreeNode

class RichTreeView(QTreeView):
    """A feature-rich tree view with filtering and context menu support."""
    
    # Signal emitted when an item is left-clicked
    left_clicked = Signal(QModelIndex)
    double_clicked = Signal(QModelIndex)

    def __init__(self, headers, parent=None):
        super().__init__(parent)
        
        # Setup model and proxy for filtering
        self.source_model = RichTreeModel(headers=headers)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.source_model)
        self.proxy_model.setRecursiveFilteringEnabled(True)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setModel(self.proxy_model)
        
        self.delegate = RichItemDelegate(self)
        self.setItemDelegate(self.delegate)

    def mousePressEvent(self, event):
        """Handle mouse press events to detect left-clicks."""
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.on_left_click(index)
        # Call the base implementation to ensure default behavior (like selection) is preserved
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.double_clicked.emit(index)
        super().mouseDoubleClickEvent(event)

    def on_left_click(self, index: QModelIndex):
        """
        This method is called when an item is left-clicked.
        It can be overridden in a subclass for custom behavior.
        The default behavior is to emit the `left_clicked` signal.
        """
        self.left_clicked.emit(index)

    def populate(self, root_node: TreeNode):
        """
        Populate the tree with data from a pre-built tree of nodes.
        The root node of the model is replaced with the provided root_node.
        """
        self.source_model.beginResetModel()
        self.source_model._root_node = root_node
        self.source_model.endResetModel()
    
    def set_filter_text(self, text: str):
        """Set the text to filter the tree view by."""
        self.proxy_model.setFilterFixedString(text)

    def _add_children(self, parent_node, children):
        """Recursively add children to the tree."""
        for child_data in children:
            child_node = TreeNode(child_data['name'], parent_node)
            parent_node.append_child(child_node)
            if 'children' in child_data:
                self._add_children(child_node, child_data['children'])
