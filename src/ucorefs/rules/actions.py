"""
UCoreFS - Rule Actions

Extensible action system for rules.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from bson import ObjectId
from loguru import logger

from src.ucorefs.models.file_record import FileRecord


class IAction(ABC):
    """
    Interface for rule actions.
    
    Actions modify file records or system state.
    """
    
    @abstractmethod
    async def execute(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        """
        Execute action on file.
        
        Args:
            file: FileRecord to act on
            context: Additional context data
            
        Returns:
            True if successful
        """
        pass


class AddTagAction(IAction):
    """Action: add tag to file."""
    
    def __init__(self, tag_id: str):
        self.tag_id = ObjectId(tag_id)
    
    async def execute(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        try:
            if self.tag_id not in file.tag_ids:
                file.tag_ids.append(self.tag_id)
                await file.save()
                logger.debug(f"Added tag to {file.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add tag: {e}")
            return False


class RemoveTagAction(IAction):
    """Action: remove tag from file."""
    
    def __init__(self, tag_id: str):
        self.tag_id = ObjectId(tag_id)
    
    async def execute(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        try:
            if self.tag_id in file.tag_ids:
                file.tag_ids.remove(self.tag_id)
                await file.save()
            return True
        except Exception as e:
            logger.error(f"Failed to remove tag: {e}")
            return False


class SetRatingAction(IAction):
    """Action: set file rating."""
    
    def __init__(self, rating: int):
        self.rating = rating
    
    async def execute(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        try:
            file.rating = self.rating
            await file.save()
            return True
        except Exception as e:
            logger.error(f"Failed to set rating: {e}")
            return False


class AddToAlbumAction(IAction):
    """Action: add file to album."""
    
    def __init__(self, album_id: str):
        self.album_id = ObjectId(album_id)
    
    async def execute(self, file: FileRecord, context: Dict[str, Any] = None) -> bool:
        try:
            if self.album_id not in file.album_ids:
                file.album_ids.append(self.album_id)
                await file.save()
            return True
        except Exception as e:
            logger.error(f"Failed to add to album: {e}")
            return False


# Action registry
ACTION_TYPES = {
    "add_tag": AddTagAction,
    "remove_tag": RemoveTagAction,
    "set_rating": SetRatingAction,
    "add_to_album": AddToAlbumAction,
}


def create_action(action_type: str, params: Dict[str, Any]) -> IAction:
    """
    Create action from type and params.
    
    Args:
        action_type: Type identifier
        params: Action parameters
        
    Returns:
        IAction instance
    """
    action_class = ACTION_TYPES.get(action_type)
    if not action_class:
        logger.warning(f"Unknown action type: {action_type}")
        return None
    
    return action_class(**params)
