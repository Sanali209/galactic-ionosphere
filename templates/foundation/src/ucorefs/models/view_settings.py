"""
UCoreFS - View Settings Model

Persistent storage for user view preferences.
Stores per-view settings like sort order, grouping, and layout mode.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId

from src.core.database.orm import CollectionRecord, Field


class ViewSettings(CollectionRecord):
    """
    Persistent view settings per user/context.
    
    Stores preferences like:
    - View mode (tree, list, card)
    - Sort field and direction
    - Grouping settings
    - Column widths
    - Filter presets
    """
    
    # Context identifier (e.g., "file_pane_left", "file_pane_right")
    context_id: str = Field(index=True)
    
    # Optional user ID for multi-user support
    user_id: Optional[str] = Field(default=None)
    
    # View mode
    view_mode: str = Field(default="list")  # tree, list, card
    
    # Sorting
    sort_field: str = Field(default="name")
    sort_ascending: bool = Field(default=True)
    
    # Grouping
    group_by: Optional[str] = Field(default=None)  # None, "date", "type", "rating"
    group_collapsed: Dict[str, bool] = Field(default_factory=dict)
    
    # Filter preset name
    filter_preset: Optional[str] = Field(default=None)
    
    # Column settings (for list/tree views)
    visible_columns: list = Field(default_factory=lambda: ["name", "size", "modified"])
    column_widths: Dict[str, int] = Field(default_factory=dict)
    
    # Card view settings
    card_size: str = Field(default="medium")  # small, medium, large
    show_ratings: bool = Field(default=True)
    show_labels: bool = Field(default=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Settings:
        name = "view_settings"
        indexes = [
            {"keys": [("context_id", 1), ("user_id", 1)], "unique": True},
        ]
    
    def __str__(self) -> str:
        return f"ViewSettings({self.context_id}): {self.view_mode}"


class ViewSettingsService:
    """
    Service for managing view settings.
    
    Usage:
        settings = await ViewSettingsService.get_or_create("file_pane_left")
        settings.sort_field = "modified"
        await settings.save()
    """
    
    @staticmethod
    async def get_or_create(
        context_id: str,
        user_id: Optional[str] = None
    ) -> ViewSettings:
        """
        Get existing settings or create default.
        
        Args:
            context_id: View context identifier
            user_id: Optional user ID
            
        Returns:
            ViewSettings instance
        """
        query = {"context_id": context_id}
        if user_id:
            query["user_id"] = user_id
        
        settings = await ViewSettings.find_one(query)
        
        if not settings:
            settings = ViewSettings(
                context_id=context_id,
                user_id=user_id
            )
            await settings.save()
        
        return settings
    
    @staticmethod
    async def update_setting(
        context_id: str,
        key: str,
        value: Any,
        user_id: Optional[str] = None
    ) -> ViewSettings:
        """
        Update a single setting.
        
        Args:
            context_id: View context
            key: Setting name
            value: Setting value
            user_id: Optional user ID
            
        Returns:
            Updated ViewSettings
        """
        settings = await ViewSettingsService.get_or_create(context_id, user_id)
        
        if hasattr(settings, key):
            setattr(settings, key, value)
            settings.updated_at = datetime.now()
            await settings.save()
        
        return settings
    
    @staticmethod
    async def get_all_contexts() -> list:
        """Get all unique context IDs."""
        settings = await ViewSettings.find().to_list()
        return list(set(s.context_id for s in settings))
