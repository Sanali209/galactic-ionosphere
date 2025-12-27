import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QMenu
from PySide6.QtCore import QModelIndex

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from SLM.pySide6Ext.RichTreeView.rich_tree_view import RichTreeView
from SLM.pySide6Ext.RichTreeView.tree_node import TreeNode
from SLM.files_db.components.fs_tag import TagRecord
from SLM.db_connection import initialize_db_connection

class CustomTreeView(RichTreeView):
    """An example of how to subclass RichTreeView to add a custom context menu."""
    def create_context_menu(self, index: QModelIndex) -> QMenu | None:
        menu = QMenu(self)
        
        source_index = self.proxy_model.mapToSource(index)
        node = source_index.internalPointer()
        
        menu.addAction(f"Custom action for {node.data}")
        menu.addAction("Another action")
        return menu
        
    def on_left_click(self, index: QModelIndex):
        """Override the left-click behavior."""
        source_index = self.proxy_model.mapToSource(index)
        node = source_index.internalPointer()
        print(f"Overridden click on: {node.data}")
        # Optionally call the base implementation to still emit the signal
        # super().on_left_click(index)

def handle_left_click(index: QModelIndex):
    """A function to handle the left_clicked signal."""
    # Note: This won't be called if on_left_click is overridden without calling super()
    print(f"Signal received for click on index: {index.data()}")

def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    
    # Initialize database connection
    initialize_db_connection()
    
    main_win = QMainWindow()
    main_win.setWindowTitle('Rich Tree View Example')
    
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    
    # Add a filter box
    filter_box = QLineEdit()
    filter_box.setPlaceholderText("Filter...")
    layout.addWidget(filter_box)
    
    # Create and populate the tree view
    headers = ["Name"]
    tree_view = CustomTreeView(headers=headers)
    
    # Populate with root tags
    root_tags = TagRecord.get_all_tags(root_tags=True)
    root_node = TreeNode("Tags")
    for tag in root_tags:
        child_node = TreeNode(tag, root_node)
        root_node.append_child(child_node)
    tree_view.populate(root_node)
    
    layout.addWidget(tree_view)
    
    # Connect the filter box to the tree view
    filter_box.textChanged.connect(tree_view.set_filter_text)
    
    # Connect to the left_clicked signal
    tree_view.left_clicked.connect(handle_left_click)
    
    main_win.setCentralWidget(central_widget)
    main_win.resize(400, 300)
    main_win.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
