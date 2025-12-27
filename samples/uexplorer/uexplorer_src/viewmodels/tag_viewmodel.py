"""
Tag ViewModel for MVVM pattern.

Provides synchronized tag data between TagTreeWidget and TagSelector.
"""
import asyncio
from typing import List, Optional
from PySide6.QtCore import QObject, Signal
from bson import ObjectId
from loguru import logger

from src.ui.mvvm.viewmodel import BaseViewModel
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag


class TagViewModel(BaseViewModel):
    """
    ViewModel for tag management.
    
    Provides:
    - Centralized tag data cache
    - Change notifications for all tag widgets
    - Async operations with Qt signals
    """
    
    # Signals for data binding
    tags_loaded = Signal(list)  # Emits list of all tags
    tag_created = Signal(object)  # Emits newly created Tag
    tag_deleted = Signal(str)  # Emits deleted tag ID
    tag_updated = Signal(object)  # Emits updated Tag
    
    def __init__(self, locator):
        super().__init__(locator)
        self.tag_manager = locator.get_system(TagManager)
        
        self._all_tags: List[Tag] = []
        self._root_tags: List[Tag] = []
        self._loading = False
    
    @property
    def all_tags(self) -> List[Tag]:
        """All tags (cached)."""
        return self._all_tags
    
    @property
    def root_tags(self) -> List[Tag]:
        """Root tags only (no parent)."""
        return self._root_tags
    
    async def load_all(self):
        """Load all tags from database."""
        if self._loading:
            return
        
        self._loading = True
        try:
            self._all_tags = await Tag.find({})
            self._root_tags = [t for t in self._all_tags if t.parent_id is None]
            self.tags_loaded.emit(self._all_tags)
            logger.debug(f"TagViewModel loaded {len(self._all_tags)} tags")
        except Exception as e:
            logger.error(f"Failed to load tags: {e}")
        finally:
            self._loading = False
    
    async def create_tag(self, name: str, parent_id: ObjectId = None) -> Optional[Tag]:
        """Create new tag and notify listeners."""
        try:
            tag = await self.tag_manager.create_tag_from_path(name)
            if tag:
                self._all_tags.append(tag)
                if tag.parent_id is None:
                    self._root_tags.append(tag)
                self.tag_created.emit(tag)
            return tag
        except Exception as e:
            logger.error(f"Failed to create tag: {e}")
            return None
    
    async def delete_tag(self, tag_id: str, recursive: bool = True) -> bool:
        """Delete tag and notify listeners."""
        try:
            success = await self.tag_manager.delete_tag(ObjectId(tag_id), recursive)
            if success:
                self._all_tags = [t for t in self._all_tags if str(t._id) != tag_id]
                self._root_tags = [t for t in self._root_tags if str(t._id) != tag_id]
                self.tag_deleted.emit(tag_id)
            return success
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
            return False
    
    async def rename_tag(self, tag_id: str, new_name: str) -> Optional[Tag]:
        """Rename tag and notify listeners."""
        try:
            tag = await Tag.get(ObjectId(tag_id))
            if tag:
                tag.name = new_name
                # Update full_path
                if tag.parent_id:
                    parent = await Tag.get(tag.parent_id)
                    if parent:
                        tag.full_path = f"{parent.full_path}/{new_name}"
                else:
                    tag.full_path = new_name
                await tag.save()
                self.tag_updated.emit(tag)
            return tag
        except Exception as e:
            logger.error(f"Failed to rename tag: {e}")
            return None
    
    def get_tag_by_id(self, tag_id: str) -> Optional[Tag]:
        """Get tag from cache by ID."""
        return next((t for t in self._all_tags if str(t._id) == tag_id), None)
    
    def get_children(self, parent_id: ObjectId = None) -> List[Tag]:
        """Get children from cache."""
        return [t for t in self._all_tags if t.parent_id == parent_id]
    
    def refresh(self):
        """Trigger async reload."""
        asyncio.ensure_future(self.load_all())
