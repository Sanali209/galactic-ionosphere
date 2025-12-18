"""
UCoreFS - Filesystem Service

Main service providing entry points API for filesystem database access.
"""
from typing import List, Optional, Type
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
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
                entry_type="fs_warning",
                data={"message": message, **data}
            )
        logger.warning(f"FSService: {message}")
    
    async def _log_error(self, message: str, data: dict) -> None:
        """Log an error to the journal."""
        if self._journal:
            await self._journal.log(
                entry_type="fs_error",
                data={"message": message, **data}
            )
        logger.error(f"FSService: {message}")
