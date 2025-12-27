"""
Dockable Directory Navigation Panel for UExplorer.

Provides filesystem directory tree navigation.
Works with DockingService (QWidget-based).
Supports include/exclude filtering.
"""
from typing import Optional, List
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QHeaderView
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
import asyncio
from pathlib import Path
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase


class DirectoryPanel(PanelBase):
    """
    Dockable directory tree navigation panel.
    
    Shows library roots and their subdirectories.
    Emits signals when directory is selected for file filtering.
    
    Signals:
        filter_changed(include_paths, exclude_paths): Emitted when filter changes
    """
    
    # Emitted when directory is selected (for filtering file pane)
    directory_selected = Signal(str, str)  # directory_id, path
    
    # Filter changed signal for unified search
    filter_changed = Signal(list, list)  # (include_paths, exclude_paths)
    
    def __init__(self, parent, locator):
        self._tree = None
        self._fs_service = None
        self._dir_cache = {}  # id -> DirectoryRecord
        self._include_dirs: set = set()  # paths to include
        self._exclude_dirs: set = set()  # paths to exclude
        super().__init__(locator, parent)
        
        # Get FSService from locator
        try:
            from src.ucorefs.services.fs_service import FSService
            self._fs_service = locator.get_system(FSService)
        except (KeyError, ImportError):
            logger.warning("FSService not available")
    
    def setup_ui(self):
        """Build panel UI with header and tree."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("📂 Directories")
        title.setStyleSheet("font-weight: bold; color: #ffffff;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Refresh button
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedSize(24, 24)
        self.btn_refresh.setToolTip("Refresh Directory Tree")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #5a7aaa;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #6a8aba; }
        """)
        self.btn_refresh.clicked.connect(self._on_refresh)
        header.addWidget(self.btn_refresh)
        
        layout.addLayout(header)
        
        # Directory tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(16)
        self._tree.setAnimated(True)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2d2d30;
                color: #e0e0e0;
                border: none;
                outline: none;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #0d6efd;
            }
            QTreeWidget::item:hover {
                background-color: #3d3d40;
            }
        """)
        
        # Connect signals
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemExpanded.connect(self._on_item_expanded)
        
        # Context menu for include/exclude
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self._tree)
        
        # Hint label
        hint = QLabel("Right-click: Include (I) / Exclude (E)")
        hint.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(hint)
        
        # Initial load
        asyncio.ensure_future(self._load_roots())
    
    @property
    def tree(self) -> QTreeWidget:
        return self._tree
    
    def on_update(self, context=None):
        """Refresh tree when panel updated."""
        asyncio.ensure_future(self._load_roots())
    
    def _on_refresh(self):
        """Refresh button clicked."""
        self._tree.clear()
        self._dir_cache.clear()
        asyncio.ensure_future(self._load_roots())
    
    async def _load_roots(self):
        """Load library roots as top-level items."""
        if not self._fs_service:
            return
        
        try:
            roots = await self._fs_service.get_roots()
            
            for root in roots:
                item = self._create_dir_item(root)
                self._tree.addTopLevelItem(item)
                
                # Add placeholder for lazy loading
                placeholder = QTreeWidgetItem(item, ["Loading..."])
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            
            logger.debug(f"Loaded {len(roots)} library roots")
            
        except Exception as e:
            logger.error(f"Failed to load directory roots: {e}")
    
    def _create_dir_item(self, dir_record) -> QTreeWidgetItem:
        """Create tree item for a directory."""
        from pathlib import Path
        
        item = QTreeWidgetItem()
        name = dir_record.name or Path(dir_record.path).name
        
        # Format name with counts
        if dir_record.is_root:
            # Show file count for roots (recursive)
            if hasattr(dir_record, 'file_count') and dir_record.file_count > 0:
                item.setText(0, f"📁 {name} ({dir_record.file_count} files)")
            else:
                item.setText(0, f"📁 {name}")
        else:
            # Show child count for subdirectories
            if hasattr(dir_record, 'child_count') and dir_record.child_count > 0:
                item.setText(0, f"📂 {name} ({dir_record.child_count})")
            else:
                item.setText(0, f"📂 {name}")
        
        item.setToolTip(0, dir_record.path)
        
        # Store directory info
        dir_id = str(dir_record._id)
        item.setData(0, Qt.ItemDataRole.UserRole, dir_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, dir_record.path)
        
        self._dir_cache[dir_id] = dir_record
        
        return item
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle directory selection and filtering."""
        dir_id = item.data(0, Qt.ItemDataRole.UserRole)
        path = item.data(0, Qt.ItemDataRole.UserRole + 1)
        
        if dir_id and path:
            logger.debug(f"Directory selected: {path}")
            # Navigation
            self.directory_selected.emit(dir_id, path)
            # Unified Search Filter inclusion
            self.toggle_include(path)
    
    def _on_item_expanded(self, item: QTreeWidgetItem):
        """Lazy load children when expanded."""
        # Check if already loaded (more than just placeholder)
        if item.childCount() == 1:
            first_child = item.child(0)
            if first_child.text(0) == "Loading...":
                # Load children
                dir_id = item.data(0, Qt.ItemDataRole.UserRole)
                asyncio.ensure_future(self._load_children(item, dir_id))
    
    async def _load_children(self, parent_item: QTreeWidgetItem, dir_id: str):
        """Load child directories for a parent item."""
        if not self._fs_service:
            return
        
        try:
            from bson import ObjectId
            
            # Get subdirectories
            subdirs = await self._fs_service.get_directories(ObjectId(dir_id))
            
            # Remove placeholder
            parent_item.takeChildren()
            
            if not subdirs:
                # Show "empty" indicator
                empty_item = QTreeWidgetItem(parent_item, ["(empty)"])
                empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
                empty_item.setForeground(0, Qt.GlobalColor.gray)
                return
            
            for subdir in subdirs:
                item = self._create_dir_item(subdir)
                parent_item.addChild(item)
                
                # Add placeholder for lazy loading if has children
                if subdir.child_count > 0:
                    placeholder = QTreeWidgetItem(item, ["Loading..."])
                    placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            
            logger.debug(f"Loaded {len(subdirs)} subdirectories")
            
        except Exception as e:
            logger.error(f"Failed to load subdirectories: {e}")
            parent_item.takeChildren()
            error_item = QTreeWidgetItem(parent_item, [f"Error: {e}"])
            error_item.setForeground(0, Qt.GlobalColor.red)
    
    # ===== Include/Exclude Filter Methods =====
    
    @property
    def include_ids(self) -> List[str]:
        """Get list of included directory paths."""
        return list(self._include_dirs)
    
    @property
    def exclude_ids(self) -> List[str]:
        """Get list of excluded directory paths."""
        return list(self._exclude_dirs)
    
    def toggle_include(self, path: str):
        """Toggle directory in include list."""
        if path in self._include_dirs:
            self._include_dirs.discard(path)
        else:
            self._include_dirs.add(path)
            self._exclude_dirs.discard(path)
        self._emit_filter_changed()
    
    def toggle_exclude(self, path: str):
        """Toggle directory in exclude list."""
        if path in self._exclude_dirs:
            self._exclude_dirs.discard(path)
        else:
            self._exclude_dirs.add(path)
            self._include_dirs.discard(path)
        self._emit_filter_changed()
    
    def _emit_filter_changed(self):
        """Emit filter changed signal."""
        include = list(self._include_dirs)
        exclude = list(self._exclude_dirs)
        logger.debug(f"Directory filter: +{len(include)}, -{len(exclude)}")
        self.filter_changed.emit(include, exclude)
    
    def get_state(self) -> dict:
        """Save panel state for persistence."""
        state = {}
        if self._include_dirs:
            state['include_dirs'] = list(self._include_dirs)
        if self._exclude_dirs:
            state['exclude_dirs'] = list(self._exclude_dirs)
        return state
    
    def set_state(self, state: dict):
        """Restore panel state from saved data."""
        if not state:
            return
        
        if 'include_dirs' in state:
            self._include_dirs = set(state['include_dirs'])
        if 'exclude_dirs' in state:
            self._exclude_dirs = set(state['exclude_dirs'])
        
        if self._include_dirs or self._exclude_dirs:
            self._emit_filter_changed()
    
    def _show_context_menu(self, position):
        """Show context menu for directory include/exclude."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        item = self._tree.itemAt(position)
        if not item:
            return
        
        path = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not path:
            return
        
        menu = QMenu(self._tree)
        
        include_action = QAction("✓ Include in Search", menu)
        include_action.setShortcut("I")
        include_action.triggered.connect(lambda: self.toggle_include(path))
        menu.addAction(include_action)
        
        exclude_action = QAction("✗ Exclude from Search", menu)
        exclude_action.setShortcut("E")
        exclude_action.triggered.connect(lambda: self.toggle_exclude(path))
        menu.addAction(exclude_action)
        
        menu.exec_(self._tree.mapToGlobal(position))
    
    def clear_filters(self):
        """Clear all include/exclude filters."""
        self._include_dirs.clear()
        self._exclude_dirs.clear()
        # Don't emit - already handled by UnifiedQueryBuilder.clear_all()
