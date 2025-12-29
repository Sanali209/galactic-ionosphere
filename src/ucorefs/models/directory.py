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
    
    async def cascade_delete(self) -> dict:
        """
        Recursively delete this directory and all its children.
        
        Prevents orphaned records by deleting all descendant directories
        and files before deleting this directory.
        
        Returns:
            dict: Statistics {directories_deleted, files_deleted}
        """
        from src.ucorefs.models.file_record import FileRecord
        from loguru import logger
        
        stats = {
            "directories_deleted": 0,
            "files_deleted": 0
        }
        
        # 1. Recursively collect all descendant directory IDs
        descendant_ids = await self._collect_descendants([self._id])
        all_dir_ids = descendant_ids + [self._id]
        
        logger.info(f"Cascade delete: found {len(all_dir_ids)} directories under {self.path}")
        
        # 2. Delete all files in these directories
        file_delete_result = await FileRecord.delete_many({
            "parent_id": {"$in": all_dir_ids}
        })
        stats["files_deleted"] = file_delete_result.deleted_count if file_delete_result else 0
        
        # 3. Delete all directories in one bulk operation
        dir_delete_result = await DirectoryRecord.delete_many({
            "_id": {"$in": all_dir_ids}
        })
        stats["directories_deleted"] = dir_delete_result.deleted_count if dir_delete_result else 0
        
        logger.info(f"Cascade deleted {stats['directories_deleted']} directories, {stats['files_deleted']} files")
        
        return stats
    
    async def _collect_descendants(self, parent_ids: list) -> list:
        """
        Recursively collect all descendant directory IDs.
        
        Uses breadth-first traversal to avoid stack overflow on deep trees.
        
        Args:
            parent_ids: List of parent ObjectIds to start from
            
        Returns:
            List of all descendant directory ObjectIds
        """
        if not parent_ids:
            return []
        
        # Find immediate children
        children = await DirectoryRecord.find({
            "parent_id": {"$in": parent_ids}
        })
        
        if not children:
            return []
        
        child_ids = [child._id for child in children]
        
        # Recursively find descendants of children
        descendants = await self._collect_descendants(child_ids)
        
        return child_ids + descendants
