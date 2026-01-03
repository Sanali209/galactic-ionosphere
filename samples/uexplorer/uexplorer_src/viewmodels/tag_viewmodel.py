"""
Tag ViewModel for MVVM pattern.

Provides synchronized tag data between TagTreeWidget and TagSelector.
"""
import asyncio
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
from bson import ObjectId
from loguru import logger

from src.ui.mvvm.viewmodel import BaseViewModel
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator


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
    
    def __init__(self, locator: "ServiceLocator") -> None:
        super().__init__(locator)
        self.tag_manager: TagManager = locator.get_system(TagManager)
        
        self._all_tags: List[Tag] = []
        self._root_tags: List[Tag] = []
        self._loading = False
        
        self.initialize_reactivity()
    
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

    # === Reactive SSOT Implementation ===

    @property
    def _event_bus(self):
        """Lazy access to EventBus."""
        from src.core.events import EventBus
        try:
            return self.locator.get_system(EventBus)
        except Exception:
            return None

    def initialize_reactivity(self):
        """Subscribe to database events."""
        bus = self._event_bus
        if bus:
             bus.subscribe("db.tags.updated", self._on_tag_updated)
             bus.subscribe("db.tags.deleted", self._on_tag_deleted)
             logger.debug("TagViewModel: Reactivity initialized")

    def _on_tag_updated(self, data: dict):
        """Handle real-time tag updates (or creation)."""
        try:
            tag_id = ObjectId(data.get("id"))
            record_data = data.get("record", {})
            
            # Check if existing
            existing = next((t for t in self._all_tags if t._id == tag_id), None)
            
            if existing:
                # Update existing
                for k, v in record_data.items():
                    if hasattr(existing, k) and k != "_id":
                        # Handle ObjectId conversion if needed
                        if k == "parent_id" and v:
                            v = ObjectId(v)
                        setattr(existing, k, v)
                
                # Check root status change
                is_root = existing.parent_id is None
                in_roots = existing in self._root_tags
                
                if is_root and not in_roots:
                    self._root_tags.append(existing)
                elif not is_root and in_roots:
                    self._root_tags.remove(existing)
                    
                self.tag_updated.emit(existing)
                logger.debug(f"TagViewModel: Tag {tag_id} updated from event")
            else:
                # New tag created externally
                # Need to hydrate full object
                # Since we have raw dict, let's try to instantiate or just reload all if complex
                # Ideally we instantiate a Tag object. 
                # For safety/simplicity in this step, we can trigger a reload or instantiate if easy.
                # Let's instantiate to avoid full reload
                try:
                    new_tag = Tag._instantiate_from_data(record_data)
                    self._all_tags.append(new_tag)
                    if new_tag.parent_id is None:
                        self._root_tags.append(new_tag)
                    self.tag_created.emit(new_tag)
                    logger.debug(f"TagViewModel: New Tag {tag_id} created from event")
                except Exception as ex:
                    logger.warning(f"Could not instantiate tag from event, triggering reload: {ex}")
                    self.refresh()
                
        except Exception as e:
             logger.error(f"Error handling tag update: {e}")

    def _on_tag_deleted(self, data: dict):
        """Handle real-time tag deletion."""
        try:
            tag_id_str = str(data.get("id"))
            
            initial_len = len(self._all_tags)
            self._all_tags = [t for t in self._all_tags if str(t._id) != tag_id_str]
            self._root_tags = [t for t in self._root_tags if str(t._id) != tag_id_str]
            
            if len(self._all_tags) != initial_len:
                self.tag_deleted.emit(tag_id_str)
                logger.debug(f"TagViewModel: Tag {tag_id_str} deleted from event")
                
        except Exception as e:
            logger.error(f"Error handling tag deletion: {e}")
