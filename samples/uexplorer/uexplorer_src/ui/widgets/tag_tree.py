"""
Tag Tree Widget for UExplorer navigation panel.

OPTIMIZED VERSION: Uses QTreeView + TagTreeModel (Virtualization + Lazy Loading).
"""
from typing import TYPE_CHECKING, Optional, Dict, List
import asyncio
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QLineEdit, QMenu, 
    QAbstractItemView, QInputDialog, QMessageBox, QPushButton
)
from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel, QModelIndex
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from loguru import logger
from bson import ObjectId

from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag
from uexplorer_src.ui.models.tag_tree_model import TagTreeModel

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class TagTreeView(QTreeView):
    """Custom TreeView to handle file drops."""
    
    files_dropped = Signal(QModelIndex, list)  # index, file_ids
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat('application/x-file-ids'):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat('application/x-file-ids'):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasFormat('application/x-file-ids'):
            data = event.mimeData().data('application/x-file-ids')
            file_ids = data.data().decode('utf-8').split(',')
            
            index = self.indexAt(event.position().toPoint())
            if index.isValid():
                self.files_dropped.emit(index, file_ids)
                event.acceptProposedAction()
        else:
            super().dropEvent(event)


class TagTreeWidget(QWidget):
    """
    Virtualized Tag Tree with Search.
    Wraps QTreeView and manages TagTreeModel.
    """
    
    # Signals matching legacy interface
    tag_selected = Signal(str)
    files_dropped_on_tag = Signal(str, list)
    include_requested = Signal(str)
    exclude_requested = Signal(str)
    
    # New signal for clicked item (index) compatibility
    itemClicked = Signal(object, int)  # Emits (QModelIndex, column)
    
    def __init__(self, locator: "ServiceLocator", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.locator = locator
        self.tag_manager: TagManager = locator.get_system(TagManager)
        
        self.setup_ui()
        self.setup_model()
        
    def setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header Toolbar (matching directory panel style)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 4, 4, 4)
        header_layout.setSpacing(4)
        
        # Search Filter
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter tags...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
        """)
        self.search_box.textChanged.connect(self._on_search_changed)
        header_layout.addWidget(self.search_box)
        
        # Refresh button
        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setToolTip("Refresh tags")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4d4d4d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5d5d5d;
            }
            QPushButton:pressed {
                background-color: #3d3d3d;
            }
        """)
        self.refresh_btn.clicked.connect(lambda: asyncio.create_task(self.refresh_tags()))
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Tree View
        self.view = TagTreeView()
        self.view.setHeaderHidden(True)
        self.view.setUniformRowHeights(True)  # Optimization
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setDragEnabled(True)
        self.view.setAcceptDrops(True)
        self.view.setDragDropMode(QAbstractItemView.DropOnly)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        
        # Stylesheet
        self.view.setStyleSheet("""
            QTreeView { 
                background: #2d2d2d; 
                color: #cccccc; 
                border: none; 
            }
            QTreeView::item:hover { background: #3d3d3d; }
            QTreeView::item:selected { background: #4d4d4d; }
        """)
        
        # Signals
        self.view.clicked.connect(self._on_view_clicked)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        self.view.files_dropped.connect(self._on_files_dropped)
        
        layout.addWidget(self.view)

    def setup_model(self):
        """Initialize model and proxy."""
        self.model = TagTreeModel(self.tag_manager)
        
        # Proxy for filtering
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setRecursiveFilteringEnabled(True) # Requires Qt 5.10+
        
        self.view.setModel(self.proxy)

    async def refresh_tags(self):
        """Refresh the model."""
        if hasattr(self, 'model'):
            self.model.refresh_roots()

    def _on_search_changed(self, text: str):
        """Filter tags."""
        self.proxy.setFilterRegularExpression(text)
        if text:
            # Expand all matches? Can be slow. 
            # With recursive filtering, it should just show matches.
            self.view.expandAll()
        else:
            self.view.collapseAll()

    def _on_view_clicked(self, index: QModelIndex):
        """Handle click."""
        # Map proxy index to source index
        source_index = self.proxy.mapToSource(index)
        
        # Get Tag ID
        tag_id = self.model.data(source_index, Qt.UserRole)
        
        if tag_id:
            # Emit legacy signal
            self.tag_selected.emit(tag_id)
            # Emit compatibility signal (passing INDEX as 'item')
            self.itemClicked.emit(index, 0)

    def _on_files_dropped(self, proxy_index: QModelIndex, file_ids: list):
        """Handle files dropped on tree view item."""
        source_index = self.proxy.mapToSource(proxy_index)
        tag_id = self.model.data(source_index, Qt.UserRole)
        
        if tag_id:
            logger.info(f"Dropped {len(file_ids)} files on tag {tag_id}")
            self.files_dropped_on_tag.emit(tag_id, file_ids)
            asyncio.ensure_future(self._apply_tag_to_files(tag_id, file_ids))

    async def _apply_tag_to_files(self, tag_id: str, file_ids: list):
        """Apply tag to dropped files."""
        try:
            from src.ucorefs.models.file_record import FileRecord
            
            tag = await Tag.get(ObjectId(tag_id))
            if not tag:
                return
            
            count = 0
            for file_id in file_ids:
                try:
                    file_record = await FileRecord.get(ObjectId(file_id))
                    if file_record:
                        if not hasattr(file_record, 'tag_ids') or file_record.tag_ids is None:
                            file_record.tag_ids = []
                        if ObjectId(tag_id) not in file_record.tag_ids:
                            file_record.tag_ids.append(ObjectId(tag_id))
                            await file_record.save()
                            count += 1
                except Exception as e:
                    logger.warning(f"Failed to tag file {file_id}: {e}")
            
            # Update tag file count
            tag.file_count = (tag.file_count or 0) + count
            await tag.save()
            
            logger.info(f"Tagged {count} files with '{tag.name}'")
            self.refresh_tags()
            
        except Exception as e:
            logger.error(f"Failed to apply tags: {e}")

    def _show_context_menu(self, position):
        """Show context menu."""
        index = self.view.indexAt(position)
        if not index.isValid():
            return
            
        source_index = self.proxy.mapToSource(index)
        tag_id = self.model.data(source_index, Qt.UserRole)
        tag_name = self.model.data(source_index, Qt.DisplayRole)
        
        if not tag_id:
            return

        menu = QMenu(self)
        
        include_action = QAction("✓ Include in Filter", self)
        include_action.triggered.connect(lambda: self.include_requested.emit(tag_id))
        menu.addAction(include_action)
        
        exclude_action = QAction("✗ Exclude from Filter", self)
        exclude_action.triggered.connect(lambda: self.exclude_requested.emit(tag_id))
        menu.addAction(exclude_action)
        
        menu.addSeparator()
        
        add_child_action = QAction("Add Child Tag", self)
        add_child_action.triggered.connect(lambda: self._add_child_tag(tag_id))
        menu.addAction(add_child_action)
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_tag(tag_id, tag_name))
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_tag(tag_id))
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        recalc_action = QAction("🔄 Recalculate Count", self)
        recalc_action.triggered.connect(lambda: self._recalculate_tag_count(tag_id))
        menu.addAction(recalc_action)
        
        menu.exec_(self.view.mapToGlobal(position))

    # === Helper Methods ===
    
    def _add_child_tag(self, parent_id):
        name, ok = QInputDialog.getText(self, "Add Child Tag", "Tag name:")
        if ok and name.strip():
            asyncio.ensure_future(self._create_child_tag(parent_id, name.strip()))

    async def _create_child_tag(self, parent_id, name):
        try:
            await self.tag_manager.create_tag(name=name, parent_id=ObjectId(parent_id))
            # Refresh specific parent? Model doesn't support fine-grained refresh yet.
            self.refresh_tags() 
        except Exception as e:
            logger.error(f"Failed to create tag: {e}")

    def _rename_tag(self, tag_id, current_name):
        # Clean current name (remove count)
        clean_name = current_name.split(" (")[0]
        new_name, ok = QInputDialog.getText(self, "Rename Tag", "New name:", text=clean_name)
        if ok and new_name.strip():
            asyncio.ensure_future(self._rename_tag_async(tag_id, new_name.strip()))
            
    async def _rename_tag_async(self, tag_id, new_name):
        try:
            tag = await Tag.get(ObjectId(tag_id))
            if tag:
                tag.name = new_name
                await tag.save()
                self.refresh_tags()
        except Exception:
            pass

    def _delete_tag(self, tag_id):
        reply = QMessageBox.question(self, "Delete Tag", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            asyncio.ensure_future(self._delete_tag_async(tag_id))

    async def _delete_tag_async(self, tag_id):
        try:
            await self.tag_manager.delete_tag(ObjectId(tag_id))
            self.refresh_tags()
        except Exception:
            pass

    def _recalculate_tag_count(self, tag_id):
        asyncio.ensure_future(self.tag_manager.update_tag_count(ObjectId(tag_id)))
        self.refresh_tags()

    # === Compatibility APIs ===
    def get_expanded_items(self):
        # Stub: Future implementation: traverse model and check expanded state
        return []

    def expand_items(self, items):
        # Stub
        pass

