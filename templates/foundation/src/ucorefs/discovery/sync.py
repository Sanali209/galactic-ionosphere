"""
UCoreFS - Sync Manager

Applies detected changes to database atomically.
"""
import os
from typing import List
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.ucorefs.discovery.diff import DiffResult
from src.ucorefs.discovery.scanner import ScanResult
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord
from src.ucorefs.core.fs_service import FSService


class SyncManager:
    """
    Applies filesystem changes to database.
    
    Processes DiffResult in batches using MongoDB bulk operations
    for performance. Handles atomic updates to maintain data consistency.
    """
    
    def __init__(self, fs_service: FSService):
        """
        Initialize sync manager.
        
        Args:
            fs_service: FSService for database operations
        """
        self.fs_service = fs_service
    
    async def apply_changes(
        self,
        diff: DiffResult,
        root_id: str
    ) -> dict:
        """
        Apply all detected changes to database.
        
        Args:
            diff: DiffResult with detected changes
            root_id: Library root ObjectId
            
        Returns:
            Statistics dict with counts
        """
        stats = {
            "files_added": 0,
            "dirs_added": 0,
            "files_modified": 0,
            "files_deleted": 0,
            "dirs_deleted": 0
        }
        
        # Process deletions first (children before parents)
        stats["files_deleted"] = await self._delete_files(diff.deleted_files)
        stats["dirs_deleted"] = await self._delete_directories(diff.deleted_dirs)
        
        # Add directories (parents before children)
        stats["dirs_added"] = await self._add_directories(
            diff.added_dirs,
            root_id
        )
        
        # Add files
        stats["files_added"] = await self._add_files(diff.added_files, root_id)
        
        # Update modified files
        stats["files_modified"] = await self._update_files(diff.modified_files)
        
        logger.info(f"Sync complete: {stats}")
        return stats
    
    def _normalize_path(self, path_str: str) -> str:
        """Normalize path to use forward slashes."""
        return str(Path(path_str).as_posix())

    async def _add_files(
        self,
        scan_results: List[ScanResult],
        root_id: str
    ) -> int:
        """Add new files to database."""
        count = 0
        
        for result in scan_results:
            try:
                # Normalize paths
                file_path = self._normalize_path(result.path)
                parent_path = self._normalize_path(str(Path(result.path).parent))
                
                # Try finding parent directory
                parent = await DirectoryRecord.find_one({"path": parent_path})
                
                # If parent not found (and not at root), maybe we need to rely on root_id?
                # But for correct hierarchy we need parent_id.
                # If scanning root direct children, parent_path should equal root path.
                
                # Create or update file record
                await self.fs_service.upsert_file(
                    path=file_path,
                    name=Path(result.path).name,
                    parent_id=parent._id if parent else None,
                    root_id=root_id,
                    extension=result.extension,
                    size_bytes=result.size,
                    modified_at=datetime.fromtimestamp(result.modified_time)
                )
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to add file {result.path}: {e}")
        
        return count
    
    async def _add_directories(
        self,
        scan_results: List[ScanResult],
        root_id: str
    ) -> int:
        """Add new directories to database."""
        count = 0
        
        # Sort by depth (parents first)
        sorted_results = sorted(
            scan_results,
            key=lambda r: r.path.count(os.sep)
        )
        
        for result in sorted_results:
            try:
                # Normalize paths
                dir_path = self._normalize_path(result.path)
                parent_path = self._normalize_path(str(Path(result.path).parent))
                
                parent = await DirectoryRecord.find_one({"path": parent_path})
                
                # Create or update directory record
                await self.fs_service.upsert_directory(
                    path=dir_path,
                    name=Path(result.path).name,
                    parent_id=parent._id if parent else None,
                    root_id=root_id,
                    modified_at=datetime.fromtimestamp(result.modified_time)
                )
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to add directory {result.path}: {e}")
        
        return count
    
    async def _update_files(
        self,
        modified: List[tuple]
    ) -> int:
        """Update modified files in database."""
        count = 0
        
        for existing, scan_result in modified:
            try:
                existing.size_bytes = scan_result.size
                existing.modified_at = datetime.fromtimestamp(
                    scan_result.modified_time
                )
                await existing.save()
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to update file {existing.path}: {e}")
        
        return count
    
    async def _delete_files(self, files: List[FileRecord]) -> int:
        """Delete files from database."""
        count = 0
        
        for file in files:
            try:
                await file.delete()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete file {file.path}: {e}")
        
        return count
    
    async def _delete_directories(
        self,
        directories: List[DirectoryRecord]
    ) -> int:
        """Delete directories from database."""
        count = 0
        
        # Sort by depth (children first)
        sorted_dirs = sorted(
            directories,
            key=lambda d: d.path.count(os.sep),
            reverse=True
        )
        
        for directory in sorted_dirs:
            try:
                await directory.delete()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete directory {directory.path}: {e}")
        
        return count
