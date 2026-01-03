"""
Tag Tree Widget for UExplorer navigation panel.

Supports:
- Hierarchical tag display
- Drag-and-drop tagging (drop files on tags)
- Context menu operations
- MVVM sync via TagViewModel
"""
import asyncio
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu, QInputDialog, QMessageBox
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent
from loguru import logger
from bson import ObjectId

from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag


class TagTreeWidget(QTreeWidget):
    """Tree widget displaying hierarchical tags from database."""
    
    tag_selected = Signal(str)  # Emits tag ID when selected
    files_dropped_on_tag = Signal(str, list)  # Emits (tag_id, list of file_ids)
    include_requested = Signal(str)  # Emits tag_id to include in filter
    exclude_requested = Signal(str)  # Emits tag_id to exclude from filter
    
    def __init__(self, locator, parent=None):
        super().__init__(parent)
        self.locator = locator
        self.tag_manager = locator.get_system(TagManager)
        
        # Try to get TagViewModel if registered
        self._viewmodel = None
        try:
            from src.viewmodels.tag_viewmodel import TagViewModel
            self._viewmodel = locator.get_system(TagViewModel)
            if self._viewmodel:
                self._viewmodel.tags_loaded.connect(self._on_tags_loaded)
                self._viewmodel.tag_created.connect(self._on_tag_created)
                self._viewmodel.tag_deleted.connect(self._on_tag_deleted)
        except (KeyError, ImportError):
            pass
        
        self.setHeaderLabel("Tags")
        self.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Left-click adds tag to filter as include
        self.itemClicked.connect(self._on_item_clicked)
        
        # Enable drag-and-drop
        self.setAcceptDrops(True)
        self.setDragEnabled(False)  # Don't drag tags themselves
        self.setDropIndicatorShown(True)
        
        # Drag throttling state - prevents UI freeze during drag
        self._last_drag_item = None
        self._drag_throttle_timer = QTimer()
        self._drag_throttle_timer.setSingleShot(True)
        self._drag_throttle_timer.setInterval(50)  # 50ms throttle
        self._drag_throttle_pending_item = None
        self._drag_throttle_timer.timeout.connect(self._apply_drag_highlight)
        
        # Tag ID to TreeWidgetItem mapping
        self._tag_items = {}
        
        # Defer tag loading until event loop is running
        # NOTE: PyMongo AsyncMongoClient requires active event loop
        QTimer.singleShot(100, self._deferred_refresh)
    
    def _deferred_refresh(self):
        """Deferred refresh called after event loop is ready."""
        try:
            asyncio.ensure_future(self.refresh_tags())
        except RuntimeError as e:
            logger.warning(f"Could not refresh tags on init: {e}")
    
    def _on_tags_loaded(self, tags):
        """Handle tags loaded from ViewModel."""
        asyncio.ensure_future(self.refresh_tags())
    
    def _on_tag_created(self, tag):
        """Handle new tag from ViewModel."""
        asyncio.ensure_future(self.refresh_tags())
    
    def _on_tag_deleted(self, tag_id):
        """Handle tag deletion from ViewModel."""
        asyncio.ensure_future(self.refresh_tags())
        
    async def refresh_tags(self):
        """Load all tags from database and populate tree."""
        try:
            self.clear()
            self._tag_items = {}
            
            # Get all root tags (no parent)
            root_tags = await self.tag_manager.get_children(None)
            
            if not root_tags:
                # Show placeholder if no tags exist
                placeholder = QTreeWidgetItem(self, ["(Drop files here to tag)"])
                placeholder.setForeground(0, Qt.gray)
                return
            
            # Build tree recursively
            for tag in root_tags:
                await self._add_tag_item(tag, None)
                
            self.expandAll()
            
        except Exception as e:
            logger.error(f"Failed to load tags: {e}")
            placeholder = QTreeWidgetItem(self, [f"Error loading tags: {str(e)}"])
            placeholder.setForeground(0, Qt.red)
    
    async def _add_tag_item(self, tag: Tag, parent_item: QTreeWidgetItem = None):
        """Add tag and its children to tree."""
        # Create item
        if parent_item:
            item = QTreeWidgetItem(parent_item, [tag.name])
        else:
            item = QTreeWidgetItem(self, [tag.name])
        
        # Store tag ID in item data
        item.setData(0, Qt.UserRole, str(tag._id))
        
        # Show file count if available
        if tag.file_count > 0:
            item.setText(0, f"{tag.name} ({tag.file_count})")
        
        # Apply color if set
        if tag.color:
            try:
                item.setForeground(0, QColor(tag.color))
            except:
                pass
        
        # Store in mapping
        self._tag_items[str(tag._id)] = item
        
        # Load children recursively
        children = await self.tag_manager.get_children(tag._id)
        for child_tag in children:
            await self._add_tag_item(child_tag, item)
    
    def _on_item_clicked(self, item, column):
        """Handle left-click - add tag to filter as include."""
        tag_id = item.data(0, Qt.UserRole)
        if tag_id:
            self.include_requested.emit(tag_id)
    
    # ==================== Drag and Drop ====================
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drops from file pane."""
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
        """Highlight target tag during drag (throttled)."""
        item = self.itemAt(event.position().toPoint())
        if item and item.data(0, Qt.UserRole):
            # Reason: Only update highlight if item changed to reduce repaints
            if item != self._last_drag_item:
                self._drag_throttle_pending_item = item
                if not self._drag_throttle_timer.isActive():
                    self._drag_throttle_timer.start()
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Clear drag state when leaving widget."""
        self._last_drag_item = None
        self._drag_throttle_pending_item = None
        self._drag_throttle_timer.stop()
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        """Handle files dropped on tag."""
        item = self.itemAt(event.position().toPoint())
        if not item:
            return
        
        tag_id = item.data(0, Qt.UserRole)
        if not tag_id:
            return
        
        # Get file IDs from mime data
        if event.mimeData().hasFormat('application/x-file-ids'):
            data = event.mimeData().data('application/x-file-ids')
            file_ids = data.data().decode('utf-8').split(',')
            
            logger.info(f"Dropped {len(file_ids)} files on tag {tag_id}")
            self.files_dropped_on_tag.emit(tag_id, file_ids)
            
            # Apply tags async
            asyncio.ensure_future(self._apply_tag_to_files(tag_id, file_ids))
            
            event.acceptProposedAction()
    
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
            await self.refresh_tags()
            
        except Exception as e:
            logger.error(f"Failed to apply tags: {e}")
    
    # ==================== Context Menu ====================
    
    def _show_context_menu(self, position):
        """Show context menu for tag operations."""
        item = self.itemAt(position)
        if not item:
            return
        
        tag_id = item.data(0, Qt.UserRole)
        if not tag_id:
            return
        
        menu = QMenu(self)
        
        # === Filter actions (top) ===
        include_action = QAction("✓ Include in Filter", self)
        include_action.setShortcut("I")
        include_action.triggered.connect(lambda: self.include_requested.emit(tag_id))
        menu.addAction(include_action)
        
        exclude_action = QAction("✗ Exclude from Filter", self)
        exclude_action.setShortcut("E")
        exclude_action.triggered.connect(lambda: self.exclude_requested.emit(tag_id))
        menu.addAction(exclude_action)
        
        menu.addSeparator()
        
        # === Tag operations ===
        add_child_action = QAction("Add Child Tag", self)
        add_child_action.triggered.connect(lambda: self._add_child_tag(tag_id))
        menu.addAction(add_child_action)
        
        menu.addSeparator()
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_tag(tag_id, item))
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_tag(tag_id))
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        # Recalculate count action
        recalc_action = QAction("🔄 Recalculate Count", self)
        recalc_action.triggered.connect(lambda: self._recalculate_tag_count(tag_id))
        menu.addAction(recalc_action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def _add_child_tag(self, parent_tag_id):
        """Show dialog to add child tag."""
        name, ok = QInputDialog.getText(self, "Add Child Tag", "Tag name:")
        if ok and name.strip():
            asyncio.ensure_future(self._create_child_tag(parent_tag_id, name.strip()))
    
    async def _create_child_tag(self, parent_id: str, name: str):
        """Create child tag."""
        try:
            await self.tag_manager.create_tag(name=name, parent_id=ObjectId(parent_id))
            await self.refresh_tags()
        except Exception as e:
            logger.error(f"Failed to create child tag: {e}")
    
    def _rename_tag(self, tag_id: str, item: QTreeWidgetItem):
        """Rename tag."""
        current_name = item.text(0).split(" (")[0]  # Remove file count suffix
        new_name, ok = QInputDialog.getText(self, "Rename Tag", "New name:", text=current_name)
        if ok and new_name.strip() and new_name.strip() != current_name:
            asyncio.ensure_future(self._rename_tag_async(tag_id, new_name.strip()))
    
    async def _rename_tag_async(self, tag_id: str, new_name: str):
        """Rename tag in database."""
        try:
            tag = await Tag.get(ObjectId(tag_id))
            if tag:
                tag.name = new_name
                await tag.save()
                await self.refresh_tags()
        except Exception as e:
            logger.error(f"Failed to rename tag: {e}")
    
    def _delete_tag(self, tag_id: str):
        """Delete tag after confirmation."""
        reply = QMessageBox.question(
            self, "Delete Tag", 
            "Are you sure you want to delete this tag?\n\nThis will NOT delete the files, only the tag.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            asyncio.ensure_future(self._delete_tag_async(tag_id))
    
    async def _delete_tag_async(self, tag_id: str):
        """Delete tag from database."""
        try:
            await self.tag_manager.delete_tag(ObjectId(tag_id))
            await self.refresh_tags()
            logger.info(f"Deleted tag: {tag_id}")
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
    
    def _recalculate_tag_count(self, tag_id: str):
        """Recalculate count for this tag."""
        asyncio.ensure_future(self._recalculate_tag_count_async(tag_id))
    
    async def _recalculate_tag_count_async(self, tag_id: str):
        """Recalculate tag count and refresh display."""
        try:
            count = await self.tag_manager.update_tag_count(ObjectId(tag_id))
            logger.info(f"Recalculated tag count: {count}")
            await self.refresh_tags()
        except Exception as e:
            logger.error(f"Failed to recalculate tag count: {e}")

