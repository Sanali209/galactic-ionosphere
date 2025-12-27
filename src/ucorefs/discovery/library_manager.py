"""
UCoreFS - Library Manager

Manages library roots and their scan settings.
"""
from typing import List, Optional
from bson import ObjectId
from loguru import logger

from src.ucorefs.models.directory import DirectoryRecord
from src.ucorefs.services.fs_service import FSService


class LibraryManager:
    """
    Manages library roots, watch lists, and blacklists.
    
    Provides methods to configure which directories are scanned
    and what file types are included/excluded.
    """
    
    def __init__(self, fs_service: FSService):
        """
        Initialize library manager.
        
        Args:
            fs_service: FSService instance for database access
        """
        self.fs_service = fs_service
    
    async def get_all_roots(self) -> List[DirectoryRecord]:
        """Get all registered library roots."""
        return await self.fs_service.get_roots()
    
    async def get_enabled_roots(self) -> List[DirectoryRecord]:
        """Get only enabled library roots for scanning."""
        all_roots = await self.get_all_roots()
        # Default to True if scan_enabled is not set (backward compatibility)
        return [r for r in all_roots if getattr(r, 'scan_enabled', True)]
    
    async def add_root(
        self,
        path: str,
        watch_extensions: Optional[List[str]] = None,
        blacklist_paths: Optional[List[str]] = None,
        enabled: bool = True
    ) -> DirectoryRecord:
        """
        Add a new library root.
        
        Args:
            path: Root directory path
            watch_extensions: File extensions to scan (e.g. ["jpg", "png"])
            blacklist_paths: Subdirectories to exclude
            enabled: Whether scanning is enabled
            
        Returns:
            Created DirectoryRecord
        """
        root = await self.fs_service.add_library_root(
            path=path,
            watch_extensions=watch_extensions,
            blacklist_paths=blacklist_paths
        )
        root.scan_enabled = enabled
        await root.save()
        
        logger.info(f"Library root added: {path}")
        return root
    
    async def update_watch_extensions(
        self,
        root_id: ObjectId,
        extensions: List[str]
    ) -> DirectoryRecord:
        """
        Update watch extensions for a library root.
        
        Args:
            root_id: Root directory ID
            extensions: New list of extensions
            
        Returns:
            Updated DirectoryRecord
        """
        root = await DirectoryRecord.get(root_id)
        if not root or not root.is_root:
            raise ValueError("Not a library root")
        
        root.watch_extensions = extensions
        await root.save()
        
        logger.info(f"Updated watch extensions for {root.path}")
        return root
    
    async def add_to_blacklist(
        self,
        root_id: ObjectId,
        paths: List[str]
    ) -> DirectoryRecord:
        """
        Add paths to blacklist for a root.
        
        Args:
            root_id: Root directory ID
            paths: Paths to exclude
            
        Returns:
            Updated DirectoryRecord
        """
        root = await DirectoryRecord.get(root_id)
        if not root or not root.is_root:
            raise ValueError("Not a library root")
        
        existing = set(root.blacklist_paths or [])
        existing.update(paths)
        root.blacklist_paths = list(existing)
        await root.save()
        
        logger.info(f"Added {len(paths)} paths to blacklist for {root.path}")
        return root
    
    async def enable_root(self, root_id: ObjectId) -> None:
        """Enable scanning for a root."""
        root = await DirectoryRecord.get(root_id)
        if root and root.is_root:
            root.scan_enabled = True
            await root.save()
            logger.info(f"Enabled root: {root.path}")
    
    async def disable_root(self, root_id: ObjectId) -> None:
        """Disable scanning for a root."""
        root = await DirectoryRecord.get(root_id)
        if root and root.is_root:
            root.scan_enabled = False
            await root.save()
            logger.info(f"Disabled root: {root.path}")
    
    def should_scan_extension(
        self,
        extension: str,
        watch_extensions: List[str]
    ) -> bool:
        """
        Check if file extension should be scanned.
        
        Args:
            extension: File extension (without dot)
            watch_extensions: List of allowed extensions
            
        Returns:
            True if should scan, False otherwise
        """
        if not watch_extensions:
            return True  # No filter means scan all
        
        return extension.lower() in [e.lower() for e in watch_extensions]
    
    def is_blacklisted(
        self,
        path: str,
        blacklist_paths: List[str]
    ) -> bool:
        """
        Check if path is blacklisted.
        
        Args:
            path: Path to check
            blacklist_paths: List of blacklisted paths
            
        Returns:
            True if blacklisted, False otherwise
        """
        if not blacklist_paths:
            return False
        
        # Check if path starts with any blacklisted path
        for blacklist in blacklist_paths:
            if path.startswith(blacklist):
                return True
        
        return False
