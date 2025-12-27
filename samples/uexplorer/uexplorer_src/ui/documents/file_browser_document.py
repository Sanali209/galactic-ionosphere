"""
FileBrowserDocument - CardView-based file browser document.

Uses Foundation's CardView + BrowseViewModel for MVVM file browsing.
Replaces the old 3-view FilePaneWidget with a single CardView.
"""
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSlider, QPushButton
)
from PySide6.QtCore import Signal, Qt
from loguru import logger
import asyncio
import uuid

from src.ui.cardview.card_view import CardView
from src.ui.cardview.card_viewmodel import CardViewModel
from src.ui.cardview.models.card_item import CardItem


class FileBrowserDocument(QWidget):
    """
    CardView-based file browser document.
    
    Features:
    - Foundation CardView for virtualized file display
    - Connects to BrowseViewModel for MVVM
    - Toolbar with sort/group/size controls
    - Supports multiple instances as document tabs
    
    Signals:
        selection_changed: Emitted when file selection changes
        content_changed: Emitted when content is modified
    """
    
    # Signals
    selection_changed = Signal(list)  # list of file IDs
    content_changed = Signal()
    
    def __init__(self, locator, viewmodel=None, title: str = "Files", parent=None):
        """
        Initialize file browser document.
        
        Args:
            locator: ServiceLocator for services
            viewmodel: BrowseViewModel instance (created if None)
            title: Document title
            parent: Parent widget
        """
        super().__init__(parent)
        self.locator = locator
        self._title = title
        self.id = str(uuid.uuid4())
        self._current_directory_id: Optional[str] = None  # Track current directory for session restore
        
        # ViewModel - create or use provided
        if viewmodel:
            self._viewmodel = viewmodel
        else:
            from uexplorer_src.viewmodels.browse_view_model import BrowseViewModel
            self._viewmodel = BrowseViewModel(self.id, locator)
        
        # CardView components
        self._card_viewmodel: Optional[CardViewModel] = None
        self._card_view: Optional[CardView] = None
        
        # Services
        self._thumbnail_service = None
        self._search_service = None
        self._init_services()
        
        # Build UI
        self._setup_ui()
        
        # Connect to BrowseViewModel
        self._connect_viewmodel()
        
        logger.debug(f"FileBrowserDocument created: {title} (ID: {self.id})")
    
    def _init_services(self):
        """Get services from locator."""
        try:
            from src.ucorefs.thumbnails.service import ThumbnailService
            self._thumbnail_service = self.locator.get_system(ThumbnailService)
        except (KeyError, ImportError):
            logger.debug("ThumbnailService not available")
        
        try:
            from src.ucorefs.search.service import SearchService
            self._search_service = self.locator.get_system(SearchService)
        except (KeyError, ImportError):
            logger.debug("SearchService not available")
    
    def _setup_ui(self):
        """Build the document UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # CardView with custom widget
        from uexplorer_src.ui.widgets.file_card_widget import FileCardWidget
        
        self._card_viewmodel = CardViewModel(self.locator)
        self._card_view = CardView(self)
        self._card_view._default_factory = FileCardWidget
        self._card_view.set_thumbnail_service(self._thumbnail_service)
        self._card_view.set_data_context(self._card_viewmodel)
        
        # Connect CardView signals
        self._card_view.selection_changed.connect(self._on_card_selection)
        self._card_view.item_double_clicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self._card_view, 1)
        
        # Empty state label
        self._empty_label = QLabel("Select a folder to browse files")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
        self._empty_label.hide()
        layout.addWidget(self._empty_label)
    
    def _create_toolbar(self) -> QWidget:
        """Create toolbar with sort/group/size controls."""
        toolbar = QWidget()
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet("""
            QWidget { background-color: #f5f5f5; border-bottom: 1px solid #ddd; }
            QLabel { color: #333; font-size: 11px; }
            QComboBox { min-width: 80px; }
        """)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        
        # Sort
        layout.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Name", "Size", "Modified", "Rating", "Type"])
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        layout.addWidget(self._sort_combo)
        
        # Sort direction button
        self._sort_dir_btn = QPushButton("↑")
        self._sort_dir_btn.setFixedSize(24, 24)
        self._sort_dir_btn.setToolTip("Toggle sort direction")
        self._sort_dir_btn.clicked.connect(self._toggle_sort_direction)
        layout.addWidget(self._sort_dir_btn)
        
        # Group
        layout.addWidget(QLabel("Group:"))
        self._group_combo = QComboBox()
        self._group_combo.addItems(["None", "Type", "Date", "Rating", "Folder"])
        self._group_combo.currentTextChanged.connect(self._on_group_changed)
        layout.addWidget(self._group_combo)
        
        layout.addStretch()
        
        # Card size slider
        layout.addWidget(QLabel("Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(120)
        self._size_slider.setMaximum(400)
        self._size_slider.setValue(200)
        self._size_slider.setFixedWidth(120)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        layout.addWidget(self._size_slider)
        
        return toolbar
    
    def _connect_viewmodel(self):
        """Connect to BrowseViewModel signals."""
        self._viewmodel.results_changed.connect(self._on_results_changed)
        self._viewmodel.loading_changed.connect(self._on_loading_changed)
        self._viewmodel.sort_changed.connect(self._on_vm_sort_changed)
        self._viewmodel.group_changed.connect(self._on_vm_group_changed)
        self._viewmodel.directory_changed.connect(self._on_vm_directory_changed)
    
    # --- ViewModel Event Handlers ---
    
    def _on_results_changed(self, results: list):
        """Handle results from ViewModel - convert to CardItems."""
        logger.info(f"📊 FileBrowserDocument received {len(results)} results")
        
        if not results:
            self._card_view.hide()
            self._empty_label.show()
            return
        
        self._empty_label.hide()
        self._card_view.show()
        
        # Convert FileRecords to CardItems
        items = []
        for record in results:
            item = self._file_to_card_item(record)
            items.append(item)
        
        # Load into CardViewModel
        asyncio.create_task(self._card_viewmodel.load_items(items))
    
    def _file_to_card_item(self, record) -> CardItem:
        """Convert FileRecord to CardItem for CardView."""
        # Format file size (FSRecord uses size_bytes)
        size_bytes = getattr(record, 'size_bytes', 0) or 0
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        # Get rating
        rating = getattr(record, 'rating', 0) or 0
        
        return CardItem(
            id=str(record._id),
            title=record.name or "Untitled",
            subtitle=size_str,
            thumbnail_path=str(record._id),  # ID for ThumbnailService
            item_type=getattr(record, 'file_type', 'file') or 'file',
            data={
                "path": record.path,
                "size": size_bytes,
                "rating": rating,
                "file_type": getattr(record, 'file_type', ''),
                "modified": str(getattr(record, 'modified_at', '')),
                "extension": getattr(record, 'extension', ''),
            }
        )
    
    def _on_loading_changed(self, loading: bool):
        """Handle loading state."""
        # Could show loading indicator
        pass
    
    def _on_vm_sort_changed(self, field: str, ascending: bool):
        """Sync sort from ViewModel to UI."""
        # Map field names to combo items
        field_map = {"name": "Name", "size": "Size", "modified_at": "Modified", 
                     "rating": "Rating", "file_type": "Type"}
        combo_text = field_map.get(field, "Name")
        self._sort_combo.blockSignals(True)
        self._sort_combo.setCurrentText(combo_text)
        self._sort_combo.blockSignals(False)
        self._sort_dir_btn.setText("↑" if ascending else "↓")
    
    def _on_vm_group_changed(self, group_by: str):
        """Sync group from ViewModel to UI."""
        group_map = {None: "None", "file_type": "Type", "date": "Date", 
                     "rating": "Rating", "folder": "Folder"}
        combo_text = group_map.get(group_by, "None")
        self._group_combo.blockSignals(True)
        self._group_combo.setCurrentText(combo_text)
        self._group_combo.blockSignals(False)

    def _on_vm_directory_changed(self, directory_id: object):
        """Handle directory change from ViewModel."""
        if not directory_id:
            return
            
        # Update local session state
        self._current_directory_id = str(directory_id)
        self._save_to_session()
        
        # Trigger search/refresh
        self._execute_search()
    
    # --- UI Event Handlers ---
    
    def _on_sort_changed(self, text: str):
        """Handle sort dropdown change."""
        field_map = {"Name": "name", "Size": "size", "Modified": "modified_at",
                     "Rating": "rating", "Type": "file_type"}
        field = field_map.get(text, "name")
        ascending = self._sort_dir_btn.text() == "↑"
        self._viewmodel.set_sort(field, ascending)
        
        # Also sort in CardViewModel
        if self._card_viewmodel:
            self._card_viewmodel.sort_by_field(field)
    
    def _toggle_sort_direction(self):
        """Toggle sort direction."""
        current = self._sort_dir_btn.text()
        ascending = current != "↑"
        self._sort_dir_btn.setText("↑" if ascending else "↓")
        
        field_map = {"Name": "name", "Size": "size", "Modified": "modified_at",
                     "Rating": "rating", "Type": "file_type"}
        field = field_map.get(self._sort_combo.currentText(), "name")
        self._viewmodel.set_sort(field, ascending)
    
    def _on_group_changed(self, text: str):
        """Handle group dropdown change."""
        group_map = {"None": None, "Type": "file_type", "Date": "date",
                     "Rating": "rating", "Folder": "folder"}
        group_by = group_map.get(text)
        self._viewmodel.set_group(group_by)
        
        # Also group in CardViewModel
        if self._card_viewmodel:
            if group_by:
                self._card_viewmodel.group_by_field(f"data.{group_by}")
            else:
                self._card_viewmodel.clear_grouping()
    
    def _on_size_changed(self, value: int):
        """Handle card size slider change."""
        height = int(value * 1.3)  # Slightly taller for file cards
        if self._card_view:
            self._card_view.set_card_size(value, height)
    
    def _on_card_selection(self, selected_ids: list):
        """Handle CardView selection changes."""
        self.selection_changed.emit(selected_ids)
        self.content_changed.emit()
    
    def _on_item_double_clicked(self, item_id: str):
        """Handle double-click on file - open image viewer if image."""
        logger.info(f"Double-click file: {item_id}")
        
        # Get item from viewmodel
        item = self._card_viewmodel.get_item_by_id(item_id) if self._card_viewmodel else None
        if not item:
            return
        
        # Get file path from data
        file_path = item.data.get("path") if item.data else None
        if not file_path:
            return
        
        # Check if it's an image
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        
        if ext in image_extensions:
            self._open_image_viewer(file_path, item.title)
        else:
            # Non-image: could open with OS default app
            import subprocess
            try:
                subprocess.Popen(['start', '', file_path], shell=True)
            except Exception as e:
                logger.error(f"Failed to open file: {e}")
    
    def _open_image_viewer(self, file_path: str, title: str):
        """Open image in ImageViewerDocument."""
        from uexplorer_src.ui.documents.image_viewer_document import ImageViewerDocument
        from src.ui.docking import DockingService
        import uuid
        
        viewer = ImageViewerDocument(file_path, title)
        doc_id = f"img_{uuid.uuid4().hex[:8]}"
        
        # Get DockingService and add document to center area
        try:
            docking = self.locator.get_system(DockingService)
            docking.add_document(doc_id, viewer, title, area="center")
            logger.info(f"Opened image viewer: {title}")
        except KeyError:
            logger.error("DockingService not available")
    
    # --- Public API ---
    
    @property
    def title(self) -> str:
        """Get document title."""
        return self._title
    
    @title.setter
    def title(self, value: str):
        """Set document title."""
        self._title = value
    
    @property
    def viewmodel(self):
        """Get the BrowseViewModel."""
        return self._viewmodel
    
    def browse_directory(self, directory_id: str):
        """
        Browse files in a directory.
        
        Args:
            directory_id: MongoDB ObjectId of directory
        """
        logger.info(f"Browsing directory: {directory_id}")
        self._current_directory_id = directory_id  # Track for session restore
        
        # Save session immediately (avoids close-time C++ deletion issues)
        self._save_to_session()
        
        from bson import ObjectId
        self._viewmodel.set_directory(ObjectId(directory_id))
        self._execute_search()
    
    def _save_to_session(self):
        """Save this document's directory to session."""
        from PySide6.QtCore import QSettings
        if not self._current_directory_id:
            return
        
        settings = QSettings("UExplorer", "Session")
        open_dirs = settings.value("open_directories", []) or []
        
        # Add if not already present
        if self._current_directory_id not in open_dirs:
            open_dirs.append(self._current_directory_id)
            settings.setValue("open_directories", open_dirs)
            logger.debug(f"Session updated: {len(open_dirs)} directories")
    
    @property
    def current_directory_id(self) -> Optional[str]:
        """Get current directory ID for session persistence."""
        return self._current_directory_id
    
    def _execute_search(self):
        """Execute search based on ViewModel state."""
        if not self._search_service:
            logger.warning("SearchService not available")
            return
        
        asyncio.create_task(self._do_search())
    
    async def _do_search(self):
        """Perform async search - query files in directory."""
        try:
            self._viewmodel.set_loading(True)
            
            from src.ucorefs.models.file_record import FileRecord
            from bson import ObjectId
            
            query_state = self._viewmodel.get_query_state()
            directory_id = query_state.get("directory")
            
            if not directory_id:
                logger.warning("No directory set for browse")
                self._viewmodel.set_results([])
                return
            
            # Ensure parent_id is ObjectId
            if isinstance(directory_id, str):
                directory_id = ObjectId(directory_id)
            
            # Build query - FileRecord is files only (no is_directory filter needed)
            query = {"parent_id": directory_id}
            
            # Add text filter if present
            search_text = query_state.get("search_text", "")
            if search_text:
                query["name"] = {"$regex": search_text, "$options": "i"}
            
            logger.debug(f"Directory query: {query}")
            
            # Query directly - FileRecord.find() returns list
            files = await FileRecord.find(query, limit=500)
            
            logger.info(f"Directory browse returned {len(files)} files")
            self._viewmodel.set_results(files)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._viewmodel.set_loading(False)
    
    def refresh(self):
        """Refresh current view."""
        self._execute_search()
    
    def get_selected_items(self) -> List[str]:
        """Get list of selected file IDs."""
        if self._card_view:
            return [item.id for item in self._card_view.get_selected_items()]
        return []
    
    def can_close(self) -> bool:
        """Document can always be closed."""
        return True
    
    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "title": self._title,
            "viewmodel_state": self._viewmodel.get_state() if self._viewmodel else {}
        }
    
    def set_state(self, state: dict):
        """Restore state from serialization."""
        self._title = state.get("title", "Files")
        if self._viewmodel and "viewmodel_state" in state:
            self._viewmodel.restore_state(state["viewmodel_state"])
