import sys
import os

# Add the parent directory of 'SLM' to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QSplitter, QMainWindow,
    QStatusBar, QMenu
)
from PySide6.QtCore import Qt, QSize, QObject, Signal, QThread, QModelIndex
from PySide6.QtGui import QPixmap, QIcon
from tqdm import tqdm

from SLM.appGlue.core import Allocator
from SLM.pySide6Ext.dialogsEx import show_string_editor
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord
from SLM.pySide6Ext.RichTreeView.rich_tree_view import RichTreeView
from SLM.pySide6Ext.RichTreeView.tree_node import TreeNode
from SLM.pySide6Ext.pySide6Q import PySide6GlueApp
from applications.image_graph_qt.file_record_find import FileSearchView

class TqdmWriter:
    def __init__(self, status_bar):
        self.status_bar = status_bar

    def write(self, msg):
        self.status_bar.showMessage(msg.strip())
        QApplication.processEvents()

    def flush(self):
        pass

class TagLoader(QObject):
    tags_loaded = Signal(list)
    finished = Signal()

    def run(self):
        """Long-running task."""
        root_tags = TagRecord.get_all_tags(root_tags=True)
        self.tags_loaded.emit(root_tags)
        self.finished.emit()

class TagTreeSearchView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tag Tree Search")
        self.layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)

        # Create the tag tree view
        self.tag_tree_view = RichTreeView(headers=["Tags"])
        self.tag_tree_view.setDragDropMode(self.tag_tree_view.DragDropMode.InternalMove)
        self.tag_tree_view.setAcceptDrops(True)
        self.tag_tree_view.setDropIndicatorShown(True)
        self.splitter.addWidget(self.tag_tree_view)
        self.tag_tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tag_tree_view.customContextMenuRequested.connect(self.on_context_menu)

        # Create the file search view
        self.file_search_view = FileSearchView()
        self.splitter.addWidget(self.file_search_view)

        self.splitter.setSizes([200, 600])

        self.start_tag_loading()
        self.tag_tree_view.double_clicked.connect(self.on_tag_clicked)

    def start_tag_loading(self):
        self.thread = QThread()
        self.worker = TagLoader()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.tags_loaded.connect(self.on_tags_loaded)

        self.thread.start()
        main_window = self.window()
        if main_window and isinstance(main_window, QMainWindow):
            main_window.statusBar().showMessage("Loading tags...")

    def on_tags_loaded(self, root_tags):
        root_node = TreeNode("Tags")
        for tag in root_tags:
            child_node = TreeNode(tag, root_node)
            root_node.append_child(child_node)
        self.tag_tree_view.populate(root_node)
        main_window = self.window()
        if main_window and isinstance(main_window, QMainWindow):
            main_window.statusBar().clearMessage()

    def on_tag_clicked(self, index):
        if not index.isValid():
            return
            
        source_index = self.tag_tree_view.proxy_model.mapToSource(index)
        node = source_index.internalPointer()
        tag_record = node.data
        
        if isinstance(tag_record, TagRecord):
            # Construct a regex query to find the exact tag
            query = f'tags REGEX "^{tag_record.fullName}$"'
            self.file_search_view.text_Query_LE.setText(query)
            self.file_search_view.find()

    def on_context_menu(self, pos):
        index = self.tag_tree_view.indexAt(pos)
        if not index.isValid():
            return

        source_index = self.tag_tree_view.proxy_model.mapToSource(index)
        node = source_index.internalPointer()
        tag_record = node.data

        if isinstance(tag_record, TagRecord):
            context_menu = QMenu(self)
            edit_action = context_menu.addAction("Edit")
            delete_action = context_menu.addAction("Delete")

            edit_action.triggered.connect(lambda: self.on_edit_tag(node))
            delete_action.triggered.connect(lambda: self.on_delete_tag(node))

            context_menu.exec_(self.tag_tree_view.mapToGlobal(pos))

    def on_edit_tag(self, node):
        tag_record: TagRecord = node.data
        string, result = show_string_editor("Edit tag", tag_record.fullName)
        if result:
            tag_record.rename(string)
            # This is a simple way to refresh the view
            self.tag_tree_view.model().layoutChanged.emit()

    def on_delete_tag(self, node):
        tag_record: TagRecord = node.data
        tag_record.delete()
        self.tag_tree_view.source_model.removeNode(node)


class TagTreeSearchApp(PySide6GlueApp):
    def __init__(self):
        super().__init__()
        self.main_view = TagTreeSearchView()
        self.set_main_widget(self.main_view)
        self._main_window.setWindowTitle("Tag Tree File Search")
        self._main_window.resize(1200, 800)
        self._main_window.setStatusBar(QStatusBar())

if __name__ == '__main__':
    # --- Configuration ---
    config = Allocator.config
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"
    Allocator.disable_module("VisionModule")
    
    # --- END Configuration ---

    app = TagTreeSearchApp()
    app.run()
