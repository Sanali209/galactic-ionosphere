"""
File Pane Widget

Browsable file pane with tree view, list view, and card grid view.
Integrates with FilterManager and SelectionManager.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QListView, QStackedWidget,
    QToolBar, QLineEdit, QPushButton, QMenu, QLabel
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
from uexplorer_src.ui.widgets.card_grid_view import CardGridView
from uexplorer_src.ui.widgets.view_mode_switcher import ViewModeSwitcher, ViewMode


class DragAwareTreeView(QTreeView):
    """
    QTreeView subclass that notifies the model when drag starts/ends.
    
    Reason: Standard QTreeView drag doesn't emit signals when drag starts.
    We need to pause thumbnail loading during drag to prevent UI freezes.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_model = None
    
    def set_file_model(self, model):
        """Set reference to FileModel for drag state control."""
        self._file_model = model
    
    def startDrag(self, supportedActions):
        """Override to notify model when drag starts."""
        if self._file_model:
            self._file_model.set_drag_state(True)
        try:
            super().startDrag(supportedActions)
        finally:
            # Drag ended (either drop or cancel)
            if self._file_model:
                self._file_model.set_drag_state(False)


class DragAwareListView(QListView):
    """
    QListView subclass that notifies the model when drag starts/ends.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_model = None
    
    def set_file_model(self, model):
        """Set reference to FileModel for drag state control."""
        self._file_model = model
    
    def startDrag(self, supportedActions):
        """Override to notify model when drag starts."""
        if self._file_model:
            self._file_model.set_drag_state(True)
        try:
            super().startDrag(supportedActions)
        finally:
            # Drag ended (either drop or cancel)
            if self._file_model:
                self._file_model.set_drag_state(False)


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
    find_similar = Signal(object)  # file_id for image→vector search
    
    def __init__(self, locator):
        """
        Initialize file pane widget.
        
        Args:
            locator: ServiceLocator instance
        """
        super().__init__()
        
        self.locator = locator
        self._model = None  # Initialize to None safe-guard
        self._viewmodel = None  # BrowseViewModel instance
        
        # Setup UI
        self.setup_ui()
        self.setup_model()
    
    def set_viewmodel(self, viewmodel):
        """
        Connect to BrowseViewModel for receiving search results.
        
        Args:
            viewmodel: BrowseViewModel instance
        """
        from loguru import logger
        self._viewmodel = viewmodel
        self._results = []  # Store results locally - EMPTY on init
        
        # Subscribe to all ViewModel signals
        viewmodel.results_changed.connect(self._on_viewmodel_results)
        viewmodel.view_mode_changed.connect(self._on_viewmodel_mode)
        viewmodel.sort_changed.connect(self._on_viewmodel_sort)
        viewmodel.group_changed.connect(self._on_viewmodel_group)
        viewmodel.loading_changed.connect(self._on_viewmodel_loading)
        
        logger.debug(f"FilePaneWidget connected to ViewModel: {viewmodel.doc_id}")
    
    def _on_viewmodel_results(self, results):
        """Handle results from ViewModel - display in current view mode."""
        logger.info(f"FilePaneWidget received {len(results)} results from ViewModel")
        self._results = results  # Store for view mode switching
        self._display_in_current_view()
    
    def _on_viewmodel_mode(self, mode: str):
        """Handle view mode change from ViewModel."""
        if mode == "tree":
            self.stack.setCurrentIndex(0)
        elif mode == "list":
            self.stack.setCurrentIndex(1)
        elif mode == "card":
            self.stack.setCurrentIndex(2)
        
        # Re-display results in new view
        self._display_in_current_view()
    
    def _on_viewmodel_sort(self, field: str, ascending: bool):
        """Handle sort change from ViewModel - apply to all views."""
        # Apply sort to FileModel (tree/list views)
        if self._model and hasattr(self._model, 'set_sort'):
            self._model.set_sort(field, ascending)
        
        # Apply sort to CardGridView
        if hasattr(self, 'view_cards') and hasattr(self.view_cards, 'set_sort'):
            self.view_cards.set_sort(field, ascending)
        
        logger.debug(f"Sort applied: {field} {'asc' if ascending else 'desc'}")
    
    def _on_viewmodel_loading(self, loading: bool):
        """Handle loading state from ViewModel."""
        # Could show loading indicator
        pass
    
    def _on_viewmodel_group(self, group_by: str):
        """Handle group change from ViewModel - apply to all views."""
        # Apply group to FileModel (tree/list views)
        if self._model and hasattr(self._model, 'set_group'):
            self._model.set_group(group_by)
        
        # Apply group to CardGridView
        if hasattr(self, 'view_cards') and hasattr(self.view_cards, 'set_group'):
            self.view_cards.set_group(group_by)
        
        logger.debug(f"Group applied: {group_by}")
    
    def _display_in_current_view(self):
        """Display stored results in current view mode."""
        if not hasattr(self, '_results'):
            return
        
        current = self.stack.currentWidget()
        
        if current == self.view_cards:
            # Card view accepts FileRecords directly
            self.view_cards.set_files(self._results)
        elif current == self.view_details or current == self.view_icons:
            # Tree/List views use FileModel.set_files
            if self._model and self._results:
                self._model.set_files(self._results)
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # View Stack
        self.stack = QStackedWidget()
        
        # 1. Details View (Tree) - use drag-aware version
        self.view_details = DragAwareTreeView()
        self.view_details.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_details.customContextMenuRequested.connect(self.show_context_menu)
        self.view_details.setSelectionMode(QTreeView.ExtendedSelection)
        self.view_details.setAlternatingRowColors(True)
        self.view_details.setSortingEnabled(True)
        self.view_details.setDragEnabled(True)
        self.view_details.setDragDropMode(QTreeView.DragOnly)  # Enable drag out
        self.view_details.setDefaultDropAction(Qt.CopyAction)
        
        # 2. Icons View (List) - use drag-aware version
        self.view_icons = DragAwareListView()
        self.view_icons.setViewMode(QListView.IconMode)
        self.view_icons.setResizeMode(QListView.Adjust)
        self.view_icons.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_icons.customContextMenuRequested.connect(self.show_context_menu)
        self.view_icons.setSelectionMode(QListView.ExtendedSelection)
        self.view_icons.setDragEnabled(True)
        self.view_icons.setDragDropMode(QListView.DragOnly)  # Enable drag out
        self.view_icons.setDefaultDropAction(Qt.CopyAction)
        self.view_icons.setSpacing(10)
        self.view_icons.setIconSize(self.get_icon_size())
        
        # 3. Card Grid View - visual card-based display
        self.view_cards = CardGridView()
        self.view_cards.card_clicked.connect(self._on_card_clicked)
        self.view_cards.card_double_clicked.connect(self._on_card_double_clicked)
        self.view_cards.selection_changed.connect(self._on_card_selection_changed)
        self.view_cards.find_similar_requested.connect(self._on_find_similar)
        
        # Set thumbnail service for card view
        if hasattr(self, 'locator') and self.locator:
            try:
                from src.ucorefs.thumbnails.service import ThumbnailService
                thumb_service = self.locator.get_system(ThumbnailService)
                if thumb_service:
                    self.view_cards.set_thumbnail_service(thumb_service)
            except Exception:
                pass  # ThumbnailService not available
        
        self.stack.addWidget(self.view_details)  # Index 0
        self.stack.addWidget(self.view_icons)    # Index 1
        self.stack.addWidget(self.view_cards)    # Index 2
        
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
        
        toolbar.addSeparator()
        
        # Sort dropdown
        from PySide6.QtWidgets import QToolButton, QMenu
        
        self.sort_button = QToolButton()
        self.sort_button.setText("Sort ▼")
        self.sort_button.setToolTip("Sort by")
        self.sort_button.setPopupMode(QToolButton.InstantPopup)
        
        sort_menu = QMenu(self.sort_button)
        sort_menu.addAction("Name").triggered.connect(lambda: self._apply_sort("name"))
        sort_menu.addAction("Size").triggered.connect(lambda: self._apply_sort("size"))
        sort_menu.addAction("Modified").triggered.connect(lambda: self._apply_sort("modified"))
        sort_menu.addAction("Rating").triggered.connect(lambda: self._apply_sort("rating"))
        sort_menu.addAction("Extension").triggered.connect(lambda: self._apply_sort("extension"))
        sort_menu.addSeparator()
        sort_menu.addAction("Ascending").triggered.connect(lambda: self._set_sort_direction(True))
        sort_menu.addAction("Descending").triggered.connect(lambda: self._set_sort_direction(False))
        self.sort_button.setMenu(sort_menu)
        toolbar.addWidget(self.sort_button)
        
        # Group dropdown
        self.group_button = QToolButton()
        self.group_button.setText("Group ▼")
        self.group_button.setToolTip("Group by")
        self.group_button.setPopupMode(QToolButton.InstantPopup)
        
        group_menu = QMenu(self.group_button)
        group_menu.addAction("None").triggered.connect(lambda: self._apply_group(None))
        group_menu.addAction("File Type").triggered.connect(lambda: self._apply_group("file_type"))
        group_menu.addAction("Rating").triggered.connect(lambda: self._apply_group("rating"))
        group_menu.addAction("Date").triggered.connect(lambda: self._apply_group("date"))
        self.group_button.setMenu(group_menu)
        toolbar.addWidget(self.group_button)
        
        toolbar.addSeparator()
        
        # View mode switcher (Tree/List/Card)
        self.view_switcher = ViewModeSwitcher()
        self.view_switcher.mode_changed.connect(self._on_view_mode_changed)
        toolbar.addWidget(self.view_switcher)
        
        return toolbar
    
    def _apply_sort(self, field: str):
        """Apply sort from toolbar."""
        if hasattr(self, '_viewmodel') and self._viewmodel:
            self._viewmodel.set_sort(field, self._viewmodel.sort_ascending)
            self.sort_button.setText(f"Sort: {field.title()} ▼")
        else:
            # Apply directly to model if no viewmodel
            if self._model and hasattr(self._model, 'set_sort'):
                self._model.set_sort(field, True)
            if hasattr(self, 'view_cards') and hasattr(self.view_cards, 'set_sort'):
                self.view_cards.set_sort(field, True)
    
    def _set_sort_direction(self, ascending: bool):
        """Set sort direction."""
        if hasattr(self, '_viewmodel') and self._viewmodel:
            self._viewmodel.set_sort(self._viewmodel.sort_field, ascending)
    
    def _apply_group(self, group_by: str = None):
        """Apply grouping from toolbar."""
        if hasattr(self, '_viewmodel') and self._viewmodel:
            self._viewmodel.set_group(group_by)
            self.group_button.setText(f"Group: {group_by.title() if group_by else 'None'} ▼")
        else:
            # Apply directly if no viewmodel
            if self._model and hasattr(self._model, 'set_group'):
                self._model.set_group(group_by)
            if hasattr(self, 'view_cards') and hasattr(self.view_cards, 'set_group'):
                self.view_cards.set_group(group_by)
    
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
            
            self.proxy_model = QSortFilterProxyModel()
            self.proxy_model.setSourceModel(self._model)
            self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
            self.proxy_model.setRecursiveFilteringEnabled(True)
            # Reason: Dynamic sort filter causes recalculation on every data change
            # which causes freezes during drag. Disable and call invalidateFilter manually.
            self.proxy_model.setDynamicSortFilter(False)
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
            
            # Connect drag-aware views to FileModel for drag state control
            self.view_details.set_file_model(self._model)
            self.view_icons.set_file_model(self._model)
            
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
        
        # Refresh card view if active
        if self.stack.currentIndex() == 2:  # Card view
            self._refresh_card_view()
        
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
        
        # Update SelectionManager for properties panel
        if hasattr(self, '_selection_manager') and self._selection_manager:
            self._selection_manager.set_selection(record_ids, source="file_pane")
    
    def on_filter_changed(self, text: str):
        """Handle filter text change."""
        self.proxy_model.setFilterFixedString(text)
        # Reason: Since setDynamicSortFilter is False, must manually invalidate
        self.proxy_model.invalidateFilter()
    
    def show_context_menu(self, position):
        """Show context menu."""
        view = self.view_details  # Use the tree view directly
        index = view.indexAt(position)
        if not index.isValid():
            return
        
        menu = QMenu(self)
        
        import sys
        from file_model import FileModel

        record_id = index.data(FileModel.IdRole)
        is_dir = index.data(FileModel.IsDirectoryRole)  # Get whether it's a directory
        
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
        
        # AI Features - only for image files
        if record_id and not is_dir:
            menu.addSeparator()
            find_similar_action = menu.addAction("🔍 Find Similar...")
            find_similar_action.triggered.connect(lambda: self.on_find_similar(record_id))
        
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
        from PySide6.QtCore import QTimer
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
                    # Defer dialog to main thread
                    QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Error", "Record not found."))
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
                
                # Defer dialog to main thread to avoid task conflict
                QTimer.singleShot(0, lambda: QMessageBox.information(self, "Properties", info))
                
            except Exception as e:
                logger.exception(f"Error showing properties: {e}")
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", f"Failed to get properties: {e}"))
                
        asyncio.ensure_future(show_props())
    
    def on_find_similar(self, record_id):
        """Find similar images using vector search."""
        from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton
        from PySide6.QtCore import QTimer
        import asyncio
        
        async def find_similar():
            try:
                from bson import ObjectId
                from src.ucorefs.models.file_record import FileRecord
                from src.ucorefs.vectors.service import VectorService
                
                # Get the source file
                source_file = await FileRecord.get(ObjectId(record_id))
                if not source_file:
                    QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Error", "File not found."))
                    return
                
                # Get VectorService
                vector_service = self.locator.get_system(VectorService)
                if not vector_service or not vector_service.is_available():
                    def show_unavail():
                        QMessageBox.warning(
                            self, "Vector Search Unavailable", 
                            "ChromaDB is not available. Vector search requires ChromaDB to be running.\n\n"
                            "To enable: pip install chromadb"
                        )
                    QTimer.singleShot(0, show_unavail)
                    return
                
                # Check if file has vector
                if not source_file.has_vector:
                    def show_no_vector():
                        QMessageBox.information(
                            self, "No Vector Data",
                            "This file doesn't have vector embeddings yet.\n\n"
                            "Run 'Generate Vectors' from the Tools menu to create embeddings."
                        )
                    QTimer.singleShot(0, show_no_vector)
                    return
                
                # For now, show placeholder - actual search would use embeddings
                name = source_file.name
                def show_result():
                    QMessageBox.information(
                        self, "Find Similar",
                        f"Searching for images similar to:\n{name}\n\n"
                        "Vector search is configured. Full implementation requires:\n"
                        "1. CLIP/BLIP embedding generation\n"
                        "2. ChromaDB index population\n\n"
                        "This feature is ready for embedding integration."
                    )
                QTimer.singleShot(0, show_result)
                
            except Exception as e:
                logger.exception(f"Error finding similar: {e}")
                err = str(e)
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", f"Failed to search: {err}"))
        
        asyncio.ensure_future(find_similar())
        
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
    
    # ==================== View Mode Handlers ====================
    
    def _on_view_mode_changed(self, mode: str):
        """Handle view mode switch."""
        if mode == "tree":
            self.stack.setCurrentIndex(0)
        elif mode == "list":
            self.stack.setCurrentIndex(1)
        elif mode == "card":
            self.stack.setCurrentIndex(2)
        
        # Re-display results in new view (uses _results from ViewModel)
        self._display_in_current_view()
        
        logger.debug(f"View mode changed to: {mode}")
    
    def _refresh_card_view(self):
        """Refresh card grid view with current model data."""
        if not self._model:
            return
        
        try:
            # Get current root index
            root_index = self.view_details.rootIndex()
            
            # Collect files from model
            files = []
            row_count = self._model.rowCount(root_index)
            
            for row in range(min(row_count, 200)):  # Limit for performance
                index = self._model.index(row, 0, root_index)
                if index.isValid():
                    is_dir = index.data(FileModel.IsDirectoryRole)
                    if not is_dir:  # Only files in card view
                        file_data = type('FileData', (), {
                            '_id': index.data(FileModel.IdRole),
                            'name': index.data(Qt.DisplayRole),
                            'rating': index.data(FileModel.RatingRole) or 0,
                            '_thumbnail_data': index.data(FileModel.ThumbnailDataRole)
                        })()
                        files.append(file_data)
            
            self.view_cards.set_files(files)
            
        except Exception as e:
            logger.error(f"Failed to refresh card view: {e}")
    
    def _on_card_clicked(self, file_id):
        """Handle card click in grid view."""
        self.selection_changed.emit([file_id])
        
        # Update selection manager if available
        if hasattr(self, '_selection_manager') and self._selection_manager:
            self._selection_manager.select_single(file_id, source="file_pane")
    
    def _on_card_double_clicked(self, file_id):
        """Handle card double-click - open file."""
        logger.info(f"Opening file from card: {file_id}")
        # TODO: Implement file opening
    
    def _on_card_selection_changed(self, file_ids):
        """Handle card selection changes."""
        self.selection_changed.emit(file_ids)
        
        if hasattr(self, '_selection_manager') and self._selection_manager:
            self._selection_manager.set_selection(file_ids, source="file_pane")
    
    def _on_find_similar(self, file_id):
        """Handle Find Similar request - emit to trigger image→vector search."""
        from loguru import logger
        logger.info(f"FilePaneWidget: Find Similar requested for {file_id}")
        self.find_similar.emit(file_id)
    
    # ==================== Manager Integration ====================
    
    def set_managers(self, filter_manager=None, selection_manager=None):
        """
        Connect to centralized UI managers.
        
        Args:
            filter_manager: FilterManager instance
            selection_manager: SelectionManager instance
        """
        self._filter_manager = filter_manager
        self._selection_manager = selection_manager
        
        if filter_manager:
            filter_manager.filter_changed.connect(self._on_filter_changed)
            filter_manager.search_changed.connect(self._on_search_filter_changed)
            logger.debug("FilePaneWidget: Connected to FilterManager")
        
        if selection_manager:
            selection_manager.selection_changed.connect(self._on_external_selection)
            logger.debug("FilePaneWidget: Connected to SelectionManager")
    
    def _on_filter_changed(self):
        """Handle filter changes from FilterManager."""
        if self._filter_manager and self._model:
            # Apply filter query to model
            query = self._filter_manager.get_mongo_query()
            if hasattr(self._model, 'set_filter_query'):
                self._model.set_filter_query(query)
            else:
                # Fallback: just refresh
                self.refresh()
    
    def _on_search_filter_changed(self, text: str):
        """Handle search text changes from FilterManager."""
        # Update local filter input to stay in sync
        if self.filter_input.text() != text:
            self.filter_input.setText(text)
    
    def _on_external_selection(self):
        """Handle selection changes from SelectionManager."""
        if not self._selection_manager:
            return
        
        # Don't update if we are the source
        if self._selection_manager.get_source() == "file_pane":
            return
        
        # Update card view selection
        selected_ids = self._selection_manager.get_selected_ids()
        if self.stack.currentIndex() == 2:  # Card view
            self.view_cards.select(selected_ids)
    
    def get_current_view_mode(self) -> str:
        """Get current view mode."""
        idx = self.stack.currentIndex()
        return ["tree", "list", "card"][idx]
    
    def set_view_mode(self, mode: str):
        """Set view mode programmatically."""
        self.view_switcher.set_mode(mode)
        self._on_view_mode_changed(mode)

