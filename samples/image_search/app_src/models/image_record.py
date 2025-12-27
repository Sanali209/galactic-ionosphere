"""
Image record ORM model.
"""
from datetime import datetime
from bson import ObjectId
from typing import Optional

from src.core.database.orm import CollectionRecord, Field, ReferenceField

class ImageRecord(CollectionRecord):
    """
    Represents a found image.
    Auto collection name: "image_records"
    """
    # _collection_name auto-generates as "image_records"
    
    url: str = Field(default="")
    thumbnail_url: str = Field(default="")
    title: str = Field(default="")
    width: int = Field(default=0)
    height: int = Field(default=0)
    source: str = Field(default="")  # Source website
    
    # Download info
    downloaded: bool = Field(default=False)
    local_path: str = Field(default="")
    download_timestamp: Optional[datetime] = Field(default=None)
    
    # Reference to search
    search_id: ObjectId = Field(default=None)
    
    def __str__(self):
        return f"Image: {self.title} ({self.width}x{self.height})"
