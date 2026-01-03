"""
Album Tree Widget for UExplorer navigation panel.

Displays albums (both manual and smart) in hierarchical tree.
"""
from typing import TYPE_CHECKING, Optional, Dict
import asyncio
from PySide6.QtWidgets import (QTreeWidget, QTreeWidgetItem, QMenu, 
                                QInputDialog, QMessageBox, QDialog,
                                QVBoxLayout, QHBoxLayout, QLabel, 
                                QLineEdit, QCheckBox, QTextEdit, QPushButton, QWidget)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor
from loguru import logger
from bson import ObjectId

from src.ucorefs.albums.models import Album

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


class AlbumTreeWidget(QTreeWidget):
    """Tree widget displaying albums from database."""
    
    album_selected = Signal(str, bool, dict)  # album_id, is_smart, query
    include_requested = Signal(str)  # Emits album_id to include in filter
    exclude_requested = Signal(str)  # Emits album_id to exclude from filter
    
    def __init__(self, locator: "ServiceLocator", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.locator: "ServiceLocator" = locator
        
        self.setHeaderLabel("Albums")
        self.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        
        # Enable drop for adding files to albums
        self.setAcceptDrops(True)
        
        # Drag throttling state - prevents UI freeze during drag
        self._last_drag_item: Optional[QTreeWidgetItem] = None
        self._drag_throttle_timer: QTimer = QTimer()
        self._drag_throttle_timer.setSingleShot(True)
        self._drag_throttle_timer.setInterval(50)  # 50ms throttle
        self._drag_throttle_pending_item: Optional[QTreeWidgetItem] = None
        self._drag_throttle_timer.timeout.connect(self._apply_drag_highlight)
        
        self._album_items: Dict[str, QTreeWidgetItem] = {}
        
        # Defer album loading until event loop is running
        # NOTE: PyMongo AsyncMongoClient requires active event loop
        QTimer.singleShot(100, self._deferred_refresh)
    
    def _deferred_refresh(self):
        """Deferred refresh called after event loop is ready."""
        try:
            asyncio.ensure_future(self.refresh_albums())
        except RuntimeError as e:
            logger.warning(f"Could not refresh albums on init: {e}")
    
    async def _get_album_count(self, album: Album) -> int:
        """
        Get real-time count for an album.
        
        For smart albums, executes the query to get current count.
        For manual albums, returns cached count.
        """
        try:
            from src.ucorefs.albums.manager import AlbumManager
            album_manager = self.locator.get_system(AlbumManager)
            if album_manager:
                return await album_manager.get_album_count(album._id)
        except Exception as e:
            logger.error(f"Failed to get album count: {e}")
        
        # Fallback to cached count
        return album.file_count if album.file_count else 0

    
    async def _get_album_count(self, album: Album) -> int:
        """
        Get real-time count for an album.
        
        For smart albums, executes the query to get current count.
        For manual albums, returns cached count.
        """
        try:
            from src.ucorefs.albums.manager import AlbumManager
            album_manager = self.locator.get_system(AlbumManager)
            if album_manager:
                return await album_manager.get_album_count(album._id)
        except Exception as e:
            logger.error(f"Failed to get album count: {e}")
        
        # Fallback to cached count
        return album.file_count if album.file_count else 0

    
    async def refresh_albums(self):
        """Load all albums from database."""
        try:
            self.clear()
            self._album_items = {}
            
            # Get root albums (no parent)
            root_albums = await Album.find({"parent_id": None})
            
            if not root_albums:
                placeholder = QTreeWidgetItem(self, ["(No albums - right-click to create)"])
                placeholder.setForeground(0, Qt.gray)
                return
            
            for album in root_albums:
                await self._add_album_item(album, None)
            
            self.expandAll()
            
        except Exception as e:
            logger.error(f"Failed to load albums: {e}")
    
    async def _add_album_item(self, album: Album, parent_item: QTreeWidgetItem = None):
        """Add album and its children to tree."""
        # Get real-time count using AlbumManager
        count = await self._get_album_count(album)
        
        # Format name with count
        name = album.name
        if album.is_smart:
            name = f"📊 {name}"
        if count > 0:
            name = f"{name} ({count})"
        
        if parent_item:
            item = QTreeWidgetItem(parent_item, [name])
        else:
            item = QTreeWidgetItem(self, [name])
        
        item.setData(0, Qt.UserRole, str(album._id))
        item.setData(0, Qt.UserRole + 1, album.is_smart)
        item.setData(0, Qt.UserRole + 2, album.smart_query)
        
        self._album_items[str(album._id)] = item
        
        # Load children
        children = await Album.find({"parent_id": album._id})
        for child in children:
            await self._add_album_item(child, item)

    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle album click - emit signal to filter files."""
        album_id = item.data(0, Qt.UserRole)
        if not album_id:
            return
        
        is_smart = item.data(0, Qt.UserRole + 1) or False
        query = item.data(0, Qt.UserRole + 2) or {}
        
        self.album_selected.emit(album_id, is_smart, query)
        
        # Unified Search Filter inclusion
        self.include_requested.emit(album_id)
    
    def _show_context_menu(self, position):
        """Show context menu."""
        menu = QMenu(self)
        
        item = self.itemAt(position)
        if item and item.data(0, Qt.UserRole):
            album_id = item.data(0, Qt.UserRole)
            
            # === Filter actions (top) ===
            include_action = QAction("✓ Include in Filter", self)
            include_action.setShortcut("I")
            include_action.triggered.connect(lambda: self.include_requested.emit(album_id))
            menu.addAction(include_action)
            
            exclude_action = QAction("✗ Exclude from Filter", self)
            exclude_action.setShortcut("E")
            exclude_action.triggered.connect(lambda: self.exclude_requested.emit(album_id))
            menu.addAction(exclude_action)
            
            menu.addSeparator()
        
        # Create album actions
        create_action = QAction("Create Album", self)
        create_action.triggered.connect(self._create_album)
        menu.addAction(create_action)
        
        create_smart_action = QAction("Create Smart Album...", self)
        create_smart_action.triggered.connect(self._create_smart_album)
        menu.addAction(create_smart_action)
        
        if item and item.data(0, Qt.UserRole):
            menu.addSeparator()
            
            album_id = item.data(0, Qt.UserRole)
            
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(lambda: self._rename_album(album_id, item))
            menu.addAction(rename_action)
            
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self._delete_album(album_id))
            menu.addAction(delete_action)
            
            menu.addSeparator()
            
            # Recalculate count action
            recalc_action = QAction("🔄 Recalculate Count", self)
            recalc_action.triggered.connect(lambda: self._recalculate_count(album_id))
            menu.addAction(recalc_action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def _create_album(self):
        """Create new manual album."""
        name, ok = QInputDialog.getText(self, "Create Album", "Album name:")
        if ok and name.strip():
            asyncio.ensure_future(self._create_album_async(name.strip(), False))
    
    def _create_smart_album(self):
        """Show smart album creation dialog."""
        dialog = SmartAlbumDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.name
            query = dialog.query
            asyncio.ensure_future(self._create_album_async(name, True, query))
    
    async def _create_album_async(self, name: str, is_smart: bool, query: dict = None):
        """Create album in database."""
        try:
            album = Album(
                name=name,
                is_smart=is_smart,
                smart_query=query or {}
            )
            await album.save()
            logger.info(f"Created album: {name}")
            await self.refresh_albums()
        except Exception as e:
            logger.error(f"Failed to create album: {e}")
    
    def _rename_album(self, album_id: str, item: QTreeWidgetItem):
        """Rename album."""
        current_name = item.text(0).split(" (")[0].replace("📊 ", "")
        new_name, ok = QInputDialog.getText(self, "Rename Album", "New name:", text=current_name)
        if ok and new_name.strip():
            asyncio.ensure_future(self._rename_album_async(album_id, new_name.strip()))
    
    async def _rename_album_async(self, album_id: str, new_name: str):
        """Rename album in database."""
        try:
            album = await Album.get(ObjectId(album_id))
            if album:
                album.name = new_name
                await album.save()
                await self.refresh_albums()
        except Exception as e:
            logger.error(f"Failed to rename album: {e}")
    
    def _delete_album(self, album_id: str):
        """Delete album."""
        reply = QMessageBox.question(
            self, "Delete Album",
            "Delete this album? Files will NOT be deleted.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            asyncio.ensure_future(self._delete_album_async(album_id))
    
    async def _delete_album_async(self, album_id: str):
        """Delete album from database."""
        try:
            album = await Album.get(ObjectId(album_id))
            if album:
                await album.delete()
                await self.refresh_albums()
        except Exception as e:
            logger.error(f"Failed to delete album: {e}") 
    
    def _recalculate_count(self, album_id: str):
        """Recalculate count for this album."""
        asyncio.ensure_future(self._recalculate_count_async(album_id))
    
    async def _recalculate_count_async(self, album_id: str):
        """Recalculate album count and refresh display."""
        try:
            from src.ucorefs.albums.manager import AlbumManager
            album_manager = self.locator.get_system(AlbumManager)
            if album_manager:
                count = await album_manager.update_album_count(ObjectId(album_id))
                logger.info(f"Recalculated album count: {count}")
                await self.refresh_albums()
        except Exception as e:
            logger.error(f"Failed to recalculate album count: {e}")

    
    # Drag and drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-file-ids'):
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def _apply_drag_highlight(self):
        """Apply throttled drag highlight (timer callback)."""
        if self._drag_throttle_pending_item:
            self.setCurrentItem(self._drag_throttle_pending_item)
            self._last_drag_item = self._drag_throttle_pending_item
            self._drag_throttle_pending_item = None
    
    def dragMoveEvent(self, event):
        """Highlight target album during drag (throttled)."""
        item = self.itemAt(event.position().toPoint())
        if item:
            is_smart = item.data(0, Qt.UserRole + 1)
            if not is_smart:  # Can only drop on non-smart albums
                # Reason: Only update highlight if item changed to reduce repaints
                if item != self._last_drag_item:
                    self._drag_throttle_pending_item = item
                    if not self._drag_throttle_timer.isActive():
                        self._drag_throttle_timer.start()
                event.acceptProposedAction()
                return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Clear drag state when leaving widget."""
        self._last_drag_item = None
        self._drag_throttle_pending_item = None
        self._drag_throttle_timer.stop()
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if not item:
            return
        
        album_id = item.data(0, Qt.UserRole)
        is_smart = item.data(0, Qt.UserRole + 1)
        
        if not album_id or is_smart:
            return  # Can't drop on smart albums
        
        if event.mimeData().hasFormat('application/x-file-ids'):
            data = event.mimeData().data('application/x-file-ids')
            file_ids = data.data().decode('utf-8').split(',')
            asyncio.ensure_future(self._add_files_to_album(album_id, file_ids))
            event.acceptProposedAction()
    
    async def _add_files_to_album(self, album_id: str, file_ids: list):
        """Add files to album using AlbumManager (bidirectional)."""
        try:
            # Use AlbumManager to maintain bidirectional relationship
            from src.ucorefs.albums.manager import AlbumManager
            album_manager = self.locator.get_system(AlbumManager)
            
            if not album_manager:
                logger.error("AlbumManager not available")
                return
            
            added = 0
            for fid in file_ids:
                oid = ObjectId(fid)
                success = await album_manager.add_file_to_album(ObjectId(album_id), oid)
                if success:
                    added += 1
            
            logger.info(f"Added {added} files to album (ID: {album_id})")
            await self.refresh_albums()
            
        except Exception as e:
            logger.error(f"Failed to add files to album: {e}")


class SmartAlbumDialog(QDialog):
    """Dialog for creating smart albums."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Smart Album")
        self.resize(400, 300)
        
        self.name = ""
        self.query = {}
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Name
        layout.addWidget(QLabel("Album Name:"))
        self.name_input = QLineEdit()
        layout.addWidget(self.name_input)
        
        layout.addSpacing(10)
        
        # Preset queries
        layout.addWidget(QLabel("Quick Presets:"))
        
        self.cb_untagged = QCheckBox("Untagged files")
        layout.addWidget(self.cb_untagged)
        
        self.cb_rated = QCheckBox("5-star rated")
        layout.addWidget(self.cb_rated)
        
        self.cb_recent = QCheckBox("Added this week")
        layout.addWidget(self.cb_recent)
        
        layout.addSpacing(10)
        
        # Custom query (JSON)
        layout.addWidget(QLabel("Custom Query (JSON):"))
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText('{"rating": {"$gte": 4}}')
        self.query_input.setMaximumHeight(80)
        layout.addWidget(self.query_input)
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(create_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_create(self):
        """Validate and accept."""
        self.name = self.name_input.text().strip()
        if not self.name:
            return
        
        # Build query from checkboxes
        query = {}
        
        if self.cb_untagged.isChecked():
            query["tag_ids"] = {"$size": 0}
        
        if self.cb_rated.isChecked():
            query["rating"] = 5
        
        if self.cb_recent.isChecked():
            from datetime import datetime, timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            query["created_at"] = {"$gte": week_ago}
        
        # Parse custom JSON if provided
        custom = self.query_input.toPlainText().strip()
        if custom:
            try:
                import json
                custom_query = json.loads(custom)
                query.update(custom_query)
            except:
                pass
        
        self.query = query
        self.accept()
