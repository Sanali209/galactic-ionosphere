"""
File Pane Widget

Browsable file pane with tree view.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QListView, QStackedWidget,
    QToolBar, QLineEdit, QPushButton, QMenu
)
from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel, QModelIndex, QPersistentModelIndex
from PySide6.QtGui import QAction
from loguru import logger

import sys
from pathlib import Path

# Add models to path
models_path = Path(__file__).parent.parent.parent / "models"
if str(models_path) not in sys.path:
    sys.path.insert(0, str(models_path))

from file_model import FileModel


class FilePaneWidget(QWidget):
    """
    File browser pane with tree view.
    
    Features:
    - Tree view of filesystem
    - Navigation toolbar
    - Quick filter
    - Context menu
    """
    
    # Signals
    selection_changed = Signal(list)  # List of selected record IDs
    
    def __init__(self, locator):
        """
        Initialize file pane widget.
        
        Args:
            locator: ServiceLocator instance
        """
        super().__init__()
        
        self.locator = locator
        self._model = None # Initialize to None safe-guard
        
        # Setup UI
        self.setup_ui()
        self.setup_model()
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # View Stack
        self.stack = QStackedWidget()
        
        # 1. Details View (Tree)
        self.view_details = QTreeView()
        self.view_details.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_details.customContextMenuRequested.connect(self.show_context_menu)
        self.view_details.setSelectionMode(QTreeView.ExtendedSelection)
        self.view_details.setAlternatingRowColors(True)
        self.view_details.setSortingEnabled(True)
        self.view_details.setDragEnabled(True)
        self.view_details.setAcceptDrops(True)
        self.view_details.setDropIndicatorShown(True)
        
        # 2. Icons View (List)
        self.view_icons = QListView()
        self.view_icons.setViewMode(QListView.IconMode)
        self.view_icons.setResizeMode(QListView.Adjust)
        self.view_icons.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_icons.customContextMenuRequested.connect(self.show_context_menu)
        self.view_icons.setSelectionMode(QListView.ExtendedSelection)
        self.view_icons.setDragEnabled(True)
        self.view_icons.setAcceptDrops(True)
        self.view_icons.setSpacing(10)
        self.view_icons.setIconSize(self.get_icon_size())
        
        self.stack.addWidget(self.view_details)
        self.stack.addWidget(self.view_icons)
        
        layout.addWidget(self.stack)
        
        # Navigation stacks
        self.back_stack = []
        self.forward_stack = []
        
        # Connect signals
        self.view_details.doubleClicked.connect(self.on_double_click)
        self.view_icons.doubleClicked.connect(self.on_double_click)
        
        # Apply dark theme styling
        self.apply_theme()
    
    def get_icon_size(self):
        """Get current icon size."""
        from PySide6.QtCore import QSize
        return QSize(64, 64)
    
    def create_toolbar(self) -> QToolBar:
        """Create navigation toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Back button
        self.back_action = QAction("←", self)
        self.back_action.setToolTip("Go back")
        self.back_action.triggered.connect(self.go_back)
        self.back_action.setEnabled(False)
        toolbar.addAction(self.back_action)
        
        # Forward button
        self.forward_action = QAction("→", self)
        self.forward_action.setToolTip("Go forward")
        self.forward_action.triggered.connect(self.go_forward)
        self.forward_action.setEnabled(False)
        toolbar.addAction(self.forward_action)
        
        # Up button
        self.up_action = QAction("↑", self)
        self.up_action.setToolTip("Go up")
        self.up_action.triggered.connect(self.go_up)
        toolbar.addAction(self.up_action)
        
        toolbar.addSeparator()
        
        # Path bar
        self.path_bar = QLineEdit()
        self.path_bar.setPlaceholderText("Current Path...")
        self.path_bar.setReadOnly(True)
        toolbar.addWidget(self.path_bar)
        
        # Filter input
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter...")
        self.filter_input.setMaximumWidth(200)
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self.on_filter_changed)
        toolbar.addWidget(self.filter_input)
        
        # Refresh button
        refresh_action = QAction("⟳", self)
        refresh_action.setToolTip("Refresh")
        refresh_action.triggered.connect(self.refresh)
        toolbar.addAction(refresh_action)
        
        return toolbar
    
    def setup_model(self):
        """Setup file model."""
        try:
            logger.info("FilePaneWidget: Setting up FileModel")
            
            # Import FileModel (lazy import to avoid circular dependencies)
            import sys
            from pathlib import Path
            models_path = Path(__file__).parent.parent.parent / "models"
            if str(models_path) not in sys.path:
                sys.path.insert(0, str(models_path))
            
            from file_model import FileModel
            
            # Create model (qasync handles event loop globally)
            self._model = FileModel(self.locator)
            
            # Proxy Model
            self.proxy_model = QSortFilterProxyModel()
            self.proxy_model.setSourceModel(self._model)
            self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
            self.proxy_model.setRecursiveFilteringEnabled(True)
            self.proxy_model.setDynamicSortFilter(True)
            self.proxy_model.setFilterKeyColumn(0) # Filter by Name
            
            # Set model on both views
            self.view_details.setModel(self.proxy_model)
            self.view_icons.setModel(self.proxy_model)
            
            # Share selection model (Icons follows Details)
            self.view_icons.setSelectionModel(self.view_details.selectionModel())
            
            # Set column widths for details
            self.view_details.setColumnWidth(0, 300)  # Name
            self.view_details.setColumnWidth(1, 100)  # Size
            self.view_details.setColumnWidth(2, 100)  # Type
            self.view_details.setColumnWidth(3, 150)  # Modified
            
            # Connect selection
            self.view_details.selectionModel().selectionChanged.connect(self.on_selection_changed)
            
            # Initial update
            self.update_navigation_state()
            
            logger.info("FilePaneWidget: FileModel setup complete")
            
        except Exception as e:
            logger.exception(f"Failed to setup file model: {e}")
            import traceback
            traceback.print_exc()


    @property
    def model(self):
        """Get file model."""
        return self._model
    
    def on_double_click(self, index):
        """Handle double click on item."""
        if not index.isValid():
            return
            
        is_dir = index.data(FileModel.IsDirectoryRole)
        
        if is_dir:
            self.navigate_to(index)
        else:
            # File action (open)
            file_path = index.data(FileModel.PathRole)
            logger.info(f"Opening file: {file_path}")
            # TODO: Implement file opening
            
    def navigate_to(self, index):
        """Navigate to directory index."""
        current_root = self.view_details.rootIndex()
        
        # Push current root to back stack as persistent index
        self.back_stack.append(QPersistentModelIndex(current_root))
        self.forward_stack.clear()
        
        self.set_root(index)
        
    def set_root(self, index):
        """Set new root index."""
        # Set new root on BOTH views
        self.view_details.setRootIndex(index)
        self.view_icons.setRootIndex(index)
        
        # Update path bar
        path = index.data(FileModel.PathRole) if index.isValid() else "Library Roots"
        self.path_bar.setText(path)
        
        # Update UI state
        self.update_navigation_state()
        
    def go_back(self):
        """Go back in history."""
        if not self.back_stack:
            return
            
        current_root = self.view_details.rootIndex()
        
        self.forward_stack.append(QPersistentModelIndex(current_root))
        
        prev_root = self.back_stack.pop()
        if prev_root.isValid() or prev_root == self.view_details.rootIndex(): # Handle root (invalid index)
             self.set_root(QModelIndex(prev_root))
        else:
             self.set_root(QModelIndex())

    def go_forward(self):
        """Go forward in history."""
        if not self.forward_stack:
            return
            
        current_root = self.view_details.rootIndex()
        
        self.back_stack.append(QPersistentModelIndex(current_root))
        
        next_root = self.forward_stack.pop()
        self.set_root(QModelIndex(next_root))
        
    def go_up(self):
        """Go to parent directory."""
        current_root = self.view_details.rootIndex()
        if not current_root.isValid():
            return
            
        parent = current_root.parent()
        self.navigate_to(parent)

    def refresh(self):
        """Refresh model."""
        import asyncio
        asyncio.ensure_future(self.model.refresh_roots())
    
    def update_navigation_state(self):
        """Update toolbar button states."""
        self.back_action.setEnabled(len(self.back_stack) > 0)
        self.forward_action.setEnabled(len(self.forward_stack) > 0)
        self.up_action.setEnabled(self.view_details.rootIndex().isValid())

    def on_selection_changed(self):
        """Handle selection change."""
        selected = self.view_details.selectionModel().selectedRows()
        record_ids = []
        
        for index in selected:
            record_id = index.data(FileModel.IdRole)
            if record_id:
                record_ids.append(record_id)
        
        self.selection_changed.emit(record_ids)
    
    def on_filter_changed(self, text: str):
        """Handle filter text change."""
        self.proxy_model.setFilterFixedString(text)
    
    def show_context_menu(self, position):
        """Show context menu."""
        view = self.current_view()
        index = view.indexAt(position)
        if not index.isValid():
            return
        
        menu = QMenu(self)
        
        import sys
        from file_model import FileModel

        record_id = index.data(FileModel.IdRole)
        
        # Actions
        open_action = menu.addAction("Open")
        if record_id and not is_dir:
             open_action.triggered.connect(lambda: self.on_open_file(index))
        else:
             open_action.triggered.connect(lambda: self.navigate_to(index))
             
        menu.addSeparator()
        
        tag_action = menu.addAction("Add Tags...")
        rate_action = menu.addAction("Set Rating...")
        tag_action.setEnabled(False) # TODO
        rate_action.setEnabled(False) # TODO
        
        menu.addSeparator()
        
        copy_action = menu.addAction("Copy")
        move_action = menu.addAction("Move...")
        delete_action = menu.addAction("Delete")
        
        copy_action.setEnabled(False) # TODO
        move_action.setEnabled(False) # TODO
        
        if record_id:
            delete_action.triggered.connect(lambda: self.on_delete_item(record_id, is_dir))
        else:
            delete_action.setEnabled(False)
        
        menu.addSeparator()
        
        props_action = menu.addAction("Properties")
        if record_id:
            props_action.triggered.connect(lambda: self.on_properties(record_id, is_dir))
        
        # Show menu
        menu.exec(view.viewport().mapToGlobal(position))
        
    def on_open_file(self, index):
        """Open file."""
        # TODO: Implement file opening logic (e.g. OS default or internal viewer)
        path = index.data(FileModel.PathRole)
        logger.info(f"Opening file: {path}")
        
    def on_properties(self, record_id, is_dir):
        """Show properties dialog."""
        from PySide6.QtWidgets import QMessageBox
        import asyncio
        
        # We need to fetch the record async
        async def show_props():
            try:
                # Import here to avoid circular deps
                from src.ucorefs.models.directory import DirectoryRecord
                from src.ucorefs.models.file_record import FileRecord
                
                record = None
                if is_dir:
                    record = await DirectoryRecord.get(record_id)
                else:
                    record = await FileRecord.get(record_id)
                
                if not record:
                    QMessageBox.warning(self, "Error", "Record not found.")
                    return
                
                # Show info
                info = f"Type: {'Directory' if is_dir else 'File'}\n"
                info += f"Name: {record.name}\n"
                info += f"Path: {record.path}\n"
                if hasattr(record, 'size_bytes'):
                    info += f"Size: {record.size_bytes} bytes\n"
                if hasattr(record, 'modified_at'):
                    info += f"Modified: {record.modified_at}\n"
                info += f"ID: {record.id}\n"
                
                QMessageBox.information(self, "Properties", info)
                
            except Exception as e:
                logger.exception(f"Error showing properties: {e}")
                QMessageBox.critical(self, "Error", f"Failed to get properties: {e}")
                
        asyncio.ensure_future(show_props())
        
    def on_delete_item(self, record_id, is_dir):
        """Delete item."""
        from PySide6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 
            "Confirm Delete",
            "Are you sure you want to delete this item? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            import asyncio
            async def delete_task():
                try:
                    from src.ucorefs.models.directory import DirectoryRecord
                    from src.ucorefs.models.file_record import FileRecord
                    
                    if is_dir:
                        record = await DirectoryRecord.get(record_id)
                        # Check if root?
                        if hasattr(record, 'is_root') and record.is_root:
                             QMessageBox.warning(self, "Cannot Delete", "Cannot delete library root from here. Use Library Settings.")
                             return
                    else:
                        record = await FileRecord.get(record_id)
                    
                    if record:
                        await record.delete()
                        logger.info(f"Deleted {record_id}")
                        # Refresh view
                        await self.model.refresh_roots() # Or refresh specific parent?
                except Exception as e:
                    logger.exception(f"Error deleting item: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
            
            asyncio.ensure_future(delete_task())
    
    def apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QTreeView, QListView {
                background-color: #2b2b2b;
                color: #cccccc;
                border: none;
                alternate-background-color: #323232;
            }
            QTreeView::item:selected, QListView::item:selected {
                background-color: #0e639c;
            }
            QTreeView::item:hover, QListView::item:hover {
                background-color: #3d3d3d;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #3d3d3d;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 3px;
            }
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 2px;
            }
        """)
