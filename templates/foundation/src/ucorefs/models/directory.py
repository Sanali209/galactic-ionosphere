"""
UCoreFS - DirectoryRecord Model

Represents a directory in the filesystem database.
"""
from typing import Optional
from bson import ObjectId

from src.core.database.orm import Field
from src.ucorefs.models.base import FSRecord


class DirectoryRecord(FSRecord):
    """
    Represents a directory in the filesystem.
    
    Extends FSRecord with directory-specific metadata.
    Auto collection name: "directory_records"
    """
    # Directory statistics
    child_count: int = Field(default=0)
    file_count: int = Field(default=0)  # Recursive count
    total_size: int = Field(default=0)  # Total size of contents
    
    # Library root settings
    is_root: bool = Field(default=False, index=True)
    
    # Watch/scan settings (for library roots)
    watch_extensions: list = Field(default_factory=list)
    blacklist_paths: list = Field(default_factory=list)
    scan_enabled: bool = Field(default=True)
    
    def __str__(self) -> str:
        return f"Directory: {self.name} ({self.child_count} items)"
