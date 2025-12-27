"""
Dockable Album Panel for UExplorer.

Works with DockingService (QWidget-based).
Demonstrates UCoreFS AlbumManager integration.
Supports include/exclude filtering.
"""
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox
)
from PySide6.QtCore import Signal
from typing import List
import asyncio
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase
from uexplorer_src.ui.widgets.album_tree import AlbumTreeWidget


class AlbumPanel(PanelBase):
    """
    Dockable album tree panel.
    
    Demonstrates UCoreFS AlbumManager with:
    - Regular albums (static file collections)
    - Smart albums (dynamic QueryBuilder-based)
    - Include/exclude filtering
    
    Signals:
        filter_changed(include_ids, exclude_ids): Emitted when album filter changes
        smart_album_selected(album_id, query_dict): Emitted when smart album selected
    """
    
    # Signal when smart album is selected (emits query for filtering)
    smart_album_selected = Signal(str, dict)  # album_id, query_dict
    
    # Filter changed signal for unified search
    filter_changed = Signal(list, list)  # (include_album_ids, exclude_album_ids)
    
    def __init__(self, parent, locator):
        self._tree = None
        self._album_manager = None
        self._include_albums: set = set()
        self._exclude_albums: set = set()
        super().__init__(locator, parent)
        
        # Get AlbumManager from locator
        try:
            from src.ucorefs.albums.manager import AlbumManager
            self._album_manager = locator.get_system(AlbumManager)
        except (KeyError, ImportError):
            logger.warning("AlbumManager not available")
    
    def setup_ui(self):
        """Build panel UI with header and tree."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with create button
        header = QHBoxLayout()
        
        title = QLabel("📁 Albums")
        title.setStyleSheet("font-weight: bold; color: #ffffff;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Clear filters button
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(20)
        self.btn_clear.setToolTip("Clear all album filters")
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #5a5a5a;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #6a6a6a; }
        """)
        self.btn_clear.clicked.connect(self._clear_filters)
        self.btn_clear.hide()  # Show only when filters active
        header.addWidget(self.btn_clear)
        
        # Create album button
        self.btn_create = QPushButton("+")
        self.btn_create.setFixedSize(24, 24)
        self.btn_create.setToolTip("Create New Album")
        self.btn_create.setStyleSheet("""
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
        self.btn_create.clicked.connect(self._on_create_album)
        header.addWidget(self.btn_create)
        
        layout.addLayout(header)
        
        # Filter status label
        self.filter_label = QLabel("")
        self.filter_label.setStyleSheet("color: #888; font-size: 11px;")
        self.filter_label.hide()
        layout.addWidget(self.filter_label)
        
        # Album tree
        self._tree = AlbumTreeWidget(self.locator)
        layout.addWidget(self._tree)
        
        # Connect include/exclude signals
        self._tree.include_requested.connect(self.toggle_include)
        self._tree.exclude_requested.connect(self.toggle_exclude)
    
    @property
    def tree(self) -> AlbumTreeWidget:
        return self._tree
    
    @property
    def include_ids(self) -> List[str]:
        """Get list of included album IDs."""
        return list(self._include_albums)
    
    @property
    def exclude_ids(self) -> List[str]:
        """Get list of excluded album IDs."""
        return list(self._exclude_albums)
    
    def toggle_include(self, album_id: str):
        """Toggle album in include list."""
        if album_id in self._include_albums:
            self._include_albums.discard(album_id)
        else:
            self._include_albums.add(album_id)
            self._exclude_albums.discard(album_id)
        self._emit_filter_changed()
    
    def toggle_exclude(self, album_id: str):
        """Toggle album in exclude list."""
        if album_id in self._exclude_albums:
            self._exclude_albums.discard(album_id)
        else:
            self._exclude_albums.add(album_id)
            self._include_albums.discard(album_id)
        self._emit_filter_changed()
    
    def _clear_filters(self):
        """Clear all album filters."""
        self._include_albums.clear()
        self._exclude_albums.clear()
        self._emit_filter_changed()
    
    def _update_ui(self):
        """Update the filter status label and clear button visibility."""
        include = list(self._include_albums)
        exclude = list(self._exclude_albums)
        
        if include or exclude:
            parts = []
            if include:
                parts.append(f"+{len(include)}")
            if exclude:
                parts.append(f"-{len(exclude)}")
            self.filter_label.setText(f"Filter: {' '.join(parts)}")
            self.filter_label.show()
            self.btn_clear.show()
        else:
            self.filter_label.hide()
            self.btn_clear.hide()

    def _emit_filter_changed(self):
        """Update UI and emit filter changed signal."""
        self._update_ui()
        include = list(self._include_albums)
        exclude = list(self._exclude_albums)
        self.filter_changed.emit(include, exclude)
    
    def on_update(self, context=None):
        """Refresh albums when panel updated."""
        if self._tree:
            asyncio.ensure_future(self._tree.refresh_albums())
    
    def _on_create_album(self):
        """Show create album dialog - demonstrates AlbumManager usage."""
        name, ok = QInputDialog.getText(
            self, "Create Album", "Album name:"
        )
        
        if ok and name.strip():
            asyncio.ensure_future(self._create_album(name.strip()))
    
    async def _create_album(self, name: str):
        """Create album using AlbumManager."""
        if not self._album_manager:
            QMessageBox.warning(self, "Error", "AlbumManager not available")
            return
        
        try:
            # Create album via AlbumManager service
            album = await self._album_manager.create_album(name)
            logger.info(f"Created album: {album.name}")
            
            # Refresh tree
            if self._tree:
                await self._tree.refresh_albums()
                
        except Exception as e:
            logger.error(f"Failed to create album: {e}")
            QMessageBox.warning(self, "Error", f"Failed to create album: {e}")
    
    def get_state(self) -> dict:
        """Save panel state for persistence."""
        state = {}
        if self._include_albums:
            state['include_albums'] = list(self._include_albums)
        if self._exclude_albums:
            state['exclude_albums'] = list(self._exclude_albums)
        return state
    
    def set_state(self, state: dict):
        """Restore panel state from saved data."""
        if not state:
            return
        
        if 'include_albums' in state:
            self._include_albums = set(state['include_albums'])
        if 'exclude_albums' in state:
            self._exclude_albums = set(state['exclude_albums'])
        
        if self._include_albums or self._exclude_albums:
            self._emit_filter_changed()
    
    def clear_filters(self):
        """Clear all include/exclude filters."""
        self._include_albums.clear()
        self._exclude_albums.clear()
        self._update_ui()
        # Don't emit - already handled by UnifiedQueryBuilder.clear_all()
