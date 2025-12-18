"""
Tag Selector Widget for UExplorer.
"""
import asyncio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                               QListWidget, QPushButton, QCompleter, QLabel)
from PySide6.QtCore import Qt, Signal, QStringListModel
from PySide6.QtGui import QIcon, QAction
from loguru import logger
from bson import ObjectId

from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag

class TagSelector(QWidget):
    """
    Widget for selecting and managing tags for a file.
    """
    tags_changed = Signal(list)  # Emits list of tag IDs
    
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
        self.tag_manager = locator.get_system(TagManager)
        
        self._selected_tags = [] # List of Tag objects
        self._all_tags = []      # Cache of all tags for completer
        
        self.init_ui()
        
        # Load available tags for auto-complete
        asyncio.ensure_future(self._load_all_tags())
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Add tag...")
        self.input_edit.returnPressed.connect(self._add_from_input)
        
        self.add_btn = QPushButton(QIcon.fromTheme("list-add"), "")
        self.add_btn.setToolTip("Add Tag")
        self.add_btn.clicked.connect(self._add_from_input)
        
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.add_btn)
        layout.addLayout(input_layout)
        
        # Auto-completer
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.input_edit.setCompleter(self.completer)
        
        # Tag list (Chip view simulator)
        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QListWidget.NoSelection)
        self.tag_list.setFocusPolicy(Qt.NoFocus)
        self.tag_list.setMaximumHeight(100)
        self.tag_list.itemDoubleClicked.connect(self._remove_tag)
        layout.addWidget(self.tag_list)
        
    async def _load_all_tags(self):
        """Load all tags for completer."""
        try:
            # Load all tags
            self._all_tags = await Tag.find({})
            # Use full_path for hierarchical display
            names = [t.full_path or t.name for t in self._all_tags]
            
            model = QStringListModel(names)
            self.completer.setModel(model)
            
        except Exception as e:
            logger.error(f"Failed to load tags: {e}")

    def set_tags(self, tag_ids: list):
        """Set currently selected tags."""
        # This needs to be async to fetch tags, but set_tags is usually sync in Qt.
        # We trigger background fetch.
        asyncio.ensure_future(self._set_tags_async(tag_ids))
        
    async def _set_tags_async(self, tag_ids: list):
        self.tag_list.clear()
        self._selected_tags = []
        
        if not tag_ids:
            return

        try:
            # Fetch tags
            tags = await Tag.find({"_id": {"$in": tag_ids}})
            for tag in tags:
                self._add_tag_ui(tag)
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")
            
    def _add_tag_ui(self, tag: Tag):
        """Add tag to UI list."""
        # Check if already in list
        if any(t._id == tag._id for t in self._selected_tags):
            return

        self._selected_tags.append(tag)
        
        item_text = f"{tag.name}"
        self.tag_list.addItem(item_text)
        
        # Add remove button? QListWidget makes this hard per item efficiently.
        # For now, double click to remove?
        item = self.tag_list.item(self.tag_list.count() - 1)
        item.setData(Qt.UserRole, str(tag._id))
        item.setToolTip("Double click to remove")
        
    def _add_from_input(self):
        text = self.input_edit.text().strip()
        if not text:
            return
            
        asyncio.ensure_future(self._process_add_tag(text))
        self.input_edit.clear()
        
    async def _process_add_tag(self, name: str):
        # Check if tag exists by name or full_path
        tag = next((t for t in self._all_tags if t.name.lower() == name.lower() or t.full_path.lower() == name.lower()), None)
        
        if not tag:
            # Create new tag - parse hierarchy if delimiter present
            try:
                tag = await self.tag_manager.create_tag_from_path(name)
                if tag:
                    self._all_tags.append(tag)
                    # Refresh completer with full paths
                    names = [t.full_path or t.name for t in self._all_tags]
                    self.completer.setModel(QStringListModel(names))
            except Exception as e:
                logger.error(f"Failed to create tag {name}: {e}")
                return
        
        self._add_tag_ui(tag)
        self._emit_change()

    def _emit_change(self):
        ids = [t._id for t in self._selected_tags]
        self.tags_changed.emit(ids)

    def _remove_tag(self, item):
        """Remove tag from file (not from database)."""
        tag_id = item.data(Qt.UserRole)
        if not tag_id:
            return
        
        # Remove from selected tags
        self._selected_tags = [t for t in self._selected_tags if str(t._id) != tag_id]
        
        # Remove from UI
        row = self.tag_list.row(item)
        self.tag_list.takeItem(row)
        
        # Emit change
        self._emit_change()
        logger.debug(f"Removed tag {tag_id} from file")
