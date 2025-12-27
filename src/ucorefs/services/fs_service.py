"""
UCoreFS - Filesystem Service

Main service providing entry points API for filesystem database access.
"""
from typing import List, Optional, Type
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.database.manager import DatabaseManager
from src.core.journal.service import JournalService
from src.ucorefs.models.base import FSRecord
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord


class FSService(BaseSystem):
    """
    Filesystem database service providing entry points API.
    
    Provides methods to:
    - Get library roots
    - Navigate directory hierarchy
    - Search files by name/path
    - CRUD operations on filesystem records
    
    Integrates with JournalService for logging file issues.
    """
    
    # Dependency declarations for topological startup order
    depends_on = [DatabaseManager, JournalService]
    
    async def initialize(self) -> None:
        """Initialize the filesystem service."""
        logger.info("FSService initializing")
        self._journal = self.locator.get_system(JournalService)
        await super().initialize()
        logger.info("FSService ready")
    
    async def shutdown(self) -> None:
        """Shutdown the filesystem service."""
        logger.info("FSService shutting down")
        await super().shutdown()
    
    # ==================== Entry Points API ====================
    
    async def get_roots(self) -> List[DirectoryRecord]:
        """
        Get all library root directories.
        
        Returns:
            List of DirectoryRecord marked as roots
        """
        return await DirectoryRecord.find({"is_root": True})
    
    async def get_children(self, dir_id: ObjectId, limit: int = None, skip: int = 0) -> List[FSRecord]:
        """
        Get all children (files and dirs) of a directory.
        
        Args:
            dir_id: Parent directory ObjectId
            limit: Maximum number of items to return (None for all)
            skip: Number of items to skip (for pagination)
            
        Returns:
            List of FSRecord (mixed FileRecord and DirectoryRecord)
        """
        # Get directories first (usually fewer)
        dirs = await DirectoryRecord.find({"parent_id": dir_id})
        
        # Calculate file skip/limit based on dirs count
        file_skip = max(0, skip - len(dirs))
        file_limit = limit - len(dirs) if limit else None
        
        if skip >= len(dirs):
            # Skip all dirs, only get files
            files = await FileRecord.find({"parent_id": dir_id}, limit=file_limit, skip=file_skip)
            return files
        else:
            # Include some dirs and some files
            dirs_to_show = dirs[skip:]
            if limit:
                dirs_to_show = dirs_to_show[:limit]
                remaining_limit = limit - len(dirs_to_show)
                if remaining_limit > 0:
                    files = await FileRecord.find({"parent_id": dir_id}, limit=remaining_limit)
                else:
                    files = []
            else:
                files = await FileRecord.find({"parent_id": dir_id})
            
            return dirs_to_show + files
    
    async def get_files(self, dir_id: ObjectId) -> List[FileRecord]:
        """
        Get only files in a directory.
        
        Args:
            dir_id: Parent directory ObjectId
            
        Returns:
            List of FileRecord
        """
        return await FileRecord.find({"parent_id": dir_id})
    
    async def get_directories(self, dir_id: ObjectId) -> List[DirectoryRecord]:
        """
        Get only subdirectories of a directory.
        
        Args:
            dir_id: Parent directory ObjectId
            
        Returns:
            List of DirectoryRecord
        """
        return await DirectoryRecord.find({"parent_id": dir_id})
    
    async def get_by_path(self, path: str) -> Optional[FSRecord]:
        """
        Find a filesystem record by its path.
        
        Args:
            path: Absolute path to search
            
        Returns:
            FileRecord or DirectoryRecord if found, None otherwise
        """
        # Try file first
        file = await FileRecord.find_one({"path": path})
        if file:
            return file
        
        # Try directory
        return await DirectoryRecord.find_one({"path": path})
    
    async def search_by_name(
        self, 
        pattern: str, 
        file_type: Optional[str] = None,
        limit: int = 100
    ) -> List[FSRecord]:
        """
        Search files and directories by name pattern.
        
        Args:
            pattern: Search pattern (supports regex)
            file_type: Optional filter by file type
            limit: Maximum results
            
        Returns:
            List of matching FSRecord
        """
        query = {"name": {"$regex": pattern, "$options": "i"}}
        
        if file_type:
            files = await FileRecord.find(
                {**query, "file_type": file_type}, 
                limit=limit
            )
            return files
        
        files = await FileRecord.find(query, limit=limit // 2)
        dirs = await DirectoryRecord.find(query, limit=limit // 2)
        return dirs + files
    
    # ==================== CRUD Operations ====================
    
    async def create_file(
        self,
        path: str,
        name: str,
        parent_id: Optional[ObjectId] = None,
        root_id: Optional[ObjectId] = None,
        **kwargs
    ) -> FileRecord:
        """
        Create a new file record.
        
        Args:
            path: Absolute file path
            name: Filename
            parent_id: Parent directory ID
            root_id: Library root ID
            **kwargs: Additional FileRecord fields
            
        Returns:
            Created FileRecord
        """
        file = FileRecord(
            path=path,
            name=name,
            parent_id=parent_id,
            root_id=root_id,
            **kwargs
        )
        
        # Log warning for zero-size files
        if file.size_bytes == 0:
            await self._log_warning(
                f"File has zero size: {path}",
                {"path": path, "name": name}
            )
        
        await file.save()
        return file
    
    async def upsert_file(
        self,
        path: str,
        name: str,
        parent_id: Optional[ObjectId] = None,
        root_id: Optional[ObjectId] = None,
        **kwargs
    ) -> FileRecord:
        """
        Create or update a file record.
        
        Args:
            path: Absolute file path
            name: Filename
            parent_id: Parent directory ID
            root_id: Library root ID
            **kwargs: Additional FileRecord fields
            
        Returns:
            Created or Updated FileRecord
        """
        existing = await FileRecord.find_one({"path": path})
        if existing:
            # Update fields
            existing.parent_id = parent_id
            existing.root_id = root_id
            for k, v in kwargs.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            await existing.save()
            return existing
            
        # Create new
        return await self.create_file(path, name, parent_id, root_id, **kwargs)

    async def create_directory(
        self,
        path: str,
        name: str,
        parent_id: Optional[ObjectId] = None,
        root_id: Optional[ObjectId] = None,
        is_root: bool = False,
        **kwargs
    ) -> DirectoryRecord:
        """
        Create a new directory record.
        """
        directory = DirectoryRecord(
            path=path,
            name=name,
            parent_id=parent_id,
            root_id=root_id,
            is_root=is_root,
            **kwargs
        )
        await directory.save()
        return directory

    async def upsert_directory(
        self,
        path: str,
        name: str,
        parent_id: Optional[ObjectId] = None,
        root_id: Optional[ObjectId] = None,
        is_root: bool = False,
        **kwargs
    ) -> DirectoryRecord:
        """
        Create or update a directory record.
        """
        existing = await DirectoryRecord.find_one({"path": path})
        if existing:
            # Update fields
            # Be careful not to unset is_root if it was False
            if is_root:
                 existing.is_root = True
            
            existing.parent_id = parent_id
            existing.root_id = root_id
            for k, v in kwargs.items():
                if hasattr(existing, k):
                     setattr(existing, k, v)
            await existing.save()
            return existing
            
        # Create new
        return await self.create_directory(path, name, parent_id, root_id, is_root, **kwargs)
    
    async def add_library_root(
        self,
        path: str,
        watch_extensions: Optional[List[str]] = None,
        blacklist_paths: Optional[List[str]] = None
    ) -> DirectoryRecord:
        """
        Add a new library root for scanning.
        
        Args:
            path: Absolute path to root directory
            watch_extensions: List of extensions to scan
            blacklist_paths: List of paths to exclude
            
        Returns:
            Created DirectoryRecord (is_root=True)
            
        Raises:
            ValueError: If root with this path already exists
        """
        import os
        from pathlib import Path
        
        # Normalize path to POSIX (forward slashes) for consistency
        path = Path(path).as_posix()
        
        # Check for duplicate path
        existing = await DirectoryRecord.find_one({"path": path, "is_root": True})
        if existing:
            raise ValueError(f"Library root already exists: {path}")
        
        name = os.path.basename(path) or path
        
        root = await self.create_directory(
            path=path,
            name=name,
            is_root=True,
            watch_extensions=watch_extensions or [],
            blacklist_paths=blacklist_paths or []
        )
        
        # Set root_id to itself for roots
        root.root_id = root._id
        await root.save()
        
        logger.info(f"Added library root: {path}")
        return root
    
    # ==================== Journal Integration ====================
    
    async def _log_warning(self, message: str, data: dict) -> None:
        """Log a warning to the journal."""
        if self._journal:
            await self._journal.log(
                level="WARNING",
                source="FSService",
                message=message,
                details=data
            )
        logger.warning(f"FSService: {message}")
    
    async def _log_error(self, message: str, data: dict) -> None:
        """Log an error to the journal."""
        if self._journal:
            await self._journal.log(
                level="ERROR",
                source="FSService",
                message=message,
                details=data
            )
        logger.error(f"FSService: {message}")
    
    # ==================== File Operations ====================
    
    async def move_file(
        self,
        file_id: ObjectId,
        new_folder: str,
        conflict_resolution: str = "rename"
    ) -> dict:
        """
        Move a file on disk and update database.
        
        WARNING: This operation attempts to maintain consistency between filesystem 
        and database, but a crash between disk move and DB update could still cause 
        inconsistency. For mission-critical operations, consider implementing a 
        transaction log or two-phase commit.
        
        IMPORTANT: When using `conflict_resolution="overwrite"`, the existing 
        destination file is permanently deleted BEFORE the move. If the subsequent 
        database update fails, rollback can restore the source file to its original 
        location, but the overwritten destination file CANNOT be recovered. Use 
        "rename" conflict resolution if data preservation is critical.
        
        Args:
            file_id: FileRecord ObjectId to move
            new_folder: Destination folder path
            conflict_resolution: How to handle conflicts:
                - "rename": Add suffix to filename (default, safest)
                - "skip": Don't move, return error
                - "overwrite": Replace existing file (WARNING: no rollback for deleted file)
                
        Returns:
            Dict with: success, new_path, old_path, error (if any)
        """
        import shutil
        from pathlib import Path
        
        result = {"success": False, "file_id": str(file_id)}
        
        try:
            # Get file record
            file = await FileRecord.get(file_id)
            if not file:
                result["error"] = "File not found in database"
                return result
            
            old_path = Path(file.path)
            if not old_path.exists():
                result["error"] = f"Source file does not exist: {old_path}"
                return result
            
            # Create destination folder if needed
            dest_folder = Path(new_folder)
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            new_path = dest_folder / file.name
            result["old_path"] = str(old_path)
            
            # Handle conflict
            if new_path.exists():
                if conflict_resolution == "skip":
                    result["error"] = "Destination file exists"
                    return result
                elif conflict_resolution == "rename":
                    new_path = self._get_unique_path(new_path)
                elif conflict_resolution == "overwrite":
                    new_path.unlink()
            
            # Move file on disk
            shutil.move(str(old_path), str(new_path))
            
            # Update database with rollback on failure
            try:
                file.path = str(new_path)
                file.name = new_path.name  # In case renamed
                await file.save()
                
                result["success"] = True
                result["new_path"] = str(new_path)
                logger.info(f"Moved file {file_id}: {old_path} -> {new_path}")
                
            except Exception as db_error:
                # Rollback: Move file back to original location
                logger.error(f"Database update failed, rolling back file move: {db_error}")
                try:
                    shutil.move(str(new_path), str(old_path))
                    await self._log_error(
                        f"Move operation rolled back due to DB failure",
                        {"file_id": str(file_id), "old_path": str(old_path), "new_path": str(new_path)}
                    )
                    result["error"] = f"Database update failed: {db_error}"
                except Exception as rollback_error:
                    # Critical: Both operations failed
                    logger.critical(f"CRITICAL: Rollback failed! File may be in inconsistent state: {rollback_error}")
                    await self._log_error(
                        f"CRITICAL: Move rollback failed - system in inconsistent state",
                        {
                            "file_id": str(file_id), 
                            "old_path": str(old_path), 
                            "new_path": str(new_path),
                            "db_error": str(db_error),
                            "rollback_error": str(rollback_error)
                        }
                    )
                    result["error"] = f"CRITICAL: Move operation failed and rollback failed. Manual intervention required."
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to move file {file_id}: {e}")
        
        return result
    
    async def copy_file(
        self,
        file_id: ObjectId,
        dest_folder: str,
        conflict_resolution: str = "rename"
    ) -> dict:
        """
        Copy a file to a new location and create database record.
        
        Args:
            file_id: Source FileRecord ObjectId
            dest_folder: Destination folder path
            conflict_resolution: How to handle conflicts
            
        Returns:
            Dict with: success, new_file_id, new_path, error (if any)
        """
        import shutil
        from pathlib import Path
        
        result = {"success": False, "source_file_id": str(file_id)}
        
        try:
            # Get source file record
            source = await FileRecord.get(file_id)
            if not source:
                result["error"] = "Source file not found"
                return result
            
            source_path = Path(source.path)
            if not source_path.exists():
                result["error"] = f"Source file does not exist: {source_path}"
                return result
            
            # Create destination folder if needed
            dest_path = Path(dest_folder)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            new_path = dest_path / source.name
            
            # Handle conflict
            if new_path.exists():
                if conflict_resolution == "skip":
                    result["error"] = "Destination file exists"
                    return result
                elif conflict_resolution == "rename":
                    new_path = self._get_unique_path(new_path)
            
            # Copy file
            shutil.copy2(str(source_path), str(new_path))
            
            # Create new database record
            new_file = await self.create_file(
                path=str(new_path),
                name=new_path.name,
                parent_id=source.parent_id,
                root_id=source.root_id,
                file_type=source.file_type,
                extension=source.extension
            )
            
            result["success"] = True
            result["new_file_id"] = str(new_file._id)
            result["new_path"] = str(new_path)
            
            logger.info(f"Copied file {file_id} -> {new_file._id}")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to copy file {file_id}: {e}")
        
        return result
    
    async def rename_file(
        self,
        file_id: ObjectId,
        new_name: str
    ) -> dict:
        """
        Rename a file in place.
        
        Args:
            file_id: FileRecord ObjectId
            new_name: New filename (without path)
            
        Returns:
            Dict with: success, new_path, old_name, error (if any)
        """
        from pathlib import Path
        
        result = {"success": False, "file_id": str(file_id)}
        
        try:
            file = await FileRecord.get(file_id)
            if not file:
                result["error"] = "File not found"
                return result
            
            old_path = Path(file.path)
            new_path = old_path.parent / new_name
            
            result["old_name"] = file.name
            
            if new_path.exists():
                result["error"] = f"File already exists: {new_name}"
                return result
            
            # Rename on disk
            old_path.rename(new_path)
            
            # Update database
            file.path = str(new_path)
            file.name = new_name
            file.extension = new_path.suffix.lstrip(".")
            await file.save()
            
            result["success"] = True
            result["new_path"] = str(new_path)
            
            logger.info(f"Renamed file {file_id}: {result['old_name']} -> {new_name}")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to rename file {file_id}: {e}")
        
        return result
    
    def _get_unique_path(self, path) -> 'Path':
        """Get unique path by adding numeric suffix."""
        from pathlib import Path
        
        path = Path(path)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        counter = 1
        new_path = path
        
        while new_path.exists():
            new_path = parent / f"{stem}_{counter}{suffix}"
            counter += 1
        
        return new_path

