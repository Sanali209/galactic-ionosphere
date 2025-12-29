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
        self._last_source_path = None  # Track source changes
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
        
        # Expand All button
        self.btn_expand = QPushButton("⊕")
        self.btn_expand.setFixedSize(24, 24)
        self.btn_expand.setToolTip("Expand All Visible")
        self.btn_expand.setStyleSheet("""
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
        self.btn_expand.clicked.connect(self._on_expand_all)
        header.addWidget(self.btn_expand)
        
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
        """Refresh tree when panel updated and check for source changes."""
        if self._fs_service:
            # Check if source path changed
            current_source = self._get_current_source_path()
            if current_source and current_source != self._last_source_path:
                logger.info(f"Source changed: {self._last_source_path} -> {current_source}")
                self._last_source_path = current_source
                asyncio.ensure_future(self._refresh_and_expand(current_source))
                return
        
        # Normal refresh
        asyncio.ensure_future(self._load_roots())
    
    def _on_refresh(self):
        """Refresh button clicked."""
        self._tree.clear()
        self._dir_cache.clear()
        # Reset source tracking to force re-expansion
        current_source = self._get_current_source_path()
        if current_source:
            self._last_source_path = current_source
            asyncio.ensure_future(self._refresh_and_expand(current_source))
        else:
            asyncio.ensure_future(self._load_roots())
    
    def _on_expand_all(self):
        """Expand All button clicked."""
        self._tree.expandAll()
        logger.debug("Expanded all visible tree nodes")
    
    def _get_current_source_path(self) -> Optional[str]:
        """Get current source path from FSService or config."""
        if not self._fs_service:
            return None
        
        # Try to get source path from FSService attributes
        source = getattr(self._fs_service, 'source_path', None) or \
                getattr(self._fs_service, '_source_path', None)
        
        if source:
            return str(source)
        
        # Fallback: get from config
        try:
            if hasattr(self.locator, 'config') and hasattr(self.locator.config, 'data'):
                config_data = self.locator.config.data
                if hasattr(config_data, 'library') and hasattr(config_data.library, 'source_path'):
                    return str(config_data.library.source_path)
        except Exception:
            pass
        
        return None
    
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
    
    async def _refresh_and_expand(self, target_path: str):
        """Refresh tree and auto-expand to target path."""
        # Load roots first
        await self._load_roots()
        
        # Find and expand path to target
        await self._expand_to_path(target_path)
    
    async def _expand_to_path(self, target_path: str):
        """Auto-expand tree to show target path."""
        from pathlib import Path
        
        target = Path(target_path)
        target_str = target.as_posix() if hasattr(target, 'as_posix') else str(target)
        
        # Find root that contains this path
        for i in range(self._tree.topLevelItemCount()):
            root_item = self._tree.topLevelItem(i)
            root_path = root_item.data(0, Qt.ItemDataRole.UserRole + 1)
            
            if not root_path:
                continue
            
            # Check if target is under this root
            if target_str.startswith(root_path) or target_str == root_path:
                logger.debug(f"Expanding root: {root_path}")
                
                # Expand root
                if not root_item.isExpanded():
                    # Trigger lazy load
                    dir_id = root_item.data(0, Qt.ItemDataRole.UserRole)
                    if dir_id:
                        await self._load_children(root_item, dir_id)
                root_item.setExpanded(True)
                
                # Walk down to target
                if target_str != root_path:
                    await self._expand_path_recursive(root_item, target_str, root_path)
                else:
                    # Target IS the root, just select it
                    self._tree.setCurrentItem(root_item)
                
                break
    
    async def _expand_path_recursive(self, parent_item: QTreeWidgetItem, 
                                     target_path: str, current_path: str):
        """Recursively expand tree nodes to reach target path."""
        # Check each child
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            child_path = child.data(0, Qt.ItemDataRole.UserRole + 1)
            
            if not child_path:
                continue
            
            # If target starts with child path, expand it
            if target_path.startswith(child_path) or target_path == child_path:
                logger.debug(f"Expanding: {child_path}")
                
                # Load children if not loaded
                if child.childCount() == 1 and child.child(0).text(0) == "Loading...":
                    dir_id = child.data(0, Qt.ItemDataRole.UserRole)
                    if dir_id:
                        await self._load_children(child, dir_id)
                
                child.setExpanded(True)
                
                # Continue down the tree if not at target yet
                if child_path != target_path:
                    await self._expand_path_recursive(child, target_path, child_path)
                else:
                    # Found target, select it
                    self._tree.setCurrentItem(child)
                
                break
    
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
            
            # Check if Ctrl is pressed - if so, add to filter instead of replacing
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import Qt as QtCore
            
            modifiers = QApplication.keyboardModifiers()
            if modifiers & QtCore.KeyboardModifier.ControlModifier:
                # Add to filter without replacing (toggle)
                self.toggle_include(path)
            else:
                # Replace entire directory filter section with this directory
                self.replace_directory_filter(path)
    
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
    
    def replace_directory_filter(self, path: str):
        """Replace all directory filters with just this directory."""
        # Clear existing directory filters
        self._include_dirs.clear()
        self._exclude_dirs.clear()
        
        # Set only this directory as included
        self._include_dirs.add(path)
        
        # Emit change
        self._emit_filter_changed()
    
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
