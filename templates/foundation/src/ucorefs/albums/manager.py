"""
UCoreFS - Album Manager

Manages albums including smart albums with dynamic queries.
"""
from typing import List, Optional, Dict, Any
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.albums.models import Album
from src.ucorefs.models.file_record import FileRecord


class AlbumManager(BaseSystem):
    """
    Album management service.
    
    Features:
    - Create/manage albums
    - Smart albums with dynamic queries
    - Add/remove files from albums
    """
    
    async def initialize(self) -> None:
        """Initialize album manager."""
        logger.info("AlbumManager initializing")
        await super().initialize()
        logger.info("AlbumManager ready")
    
    async def shutdown(self) -> None:
        """Shutdown album manager."""
        logger.info("AlbumManager shutting down")
        await super().shutdown()
    
    async def create_album(
        self,
        name: str,
        description: str = "",
        parent_id: Optional[ObjectId] = None,
        is_smart: bool = False,
        smart_query: Optional[Dict[str, Any]] = None
    ) -> Album:
        """
        Create a new album.
        
        Args:
            name: Album name
            description: Album description
            parent_id: Parent album ID
            is_smart: Whether this is a smart album
            smart_query: Query for smart album
            
        Returns:
            Created Album
        """
        album = Album(
            name=name,
            description=description,
            parent_id=parent_id,
            is_smart=is_smart,
            smart_query=smart_query or {}
        )
        
        await album.save()
        logger.info(f"Created album: {name} (smart: {is_smart})")
        
        return album
    
    async def add_file_to_album(
        self,
        album_id: ObjectId,
        file_id: ObjectId
    ) -> bool:
        """
        Add file to manual album.
        
        Args:
            album_id: Album ObjectId
            file_id: File ObjectId
            
        Returns:
            True if successful
        """
        try:
            album = await Album.get(album_id)
            if not album or album.is_smart:
                logger.warning("Cannot add files to smart albums")
                return False
            
            if file_id not in album.file_ids:
                album.file_ids.append(file_id)
                album.file_count = len(album.file_ids)
                await album.save()
                
                logger.debug(f"Added file to album {album.name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add file to album: {e}")
            return False
    
    async def remove_file_from_album(
        self,
        album_id: ObjectId,
        file_id: ObjectId
    ) -> bool:
        """
        Remove file from manual album.
        
        Args:
            album_id: Album ObjectId
            file_id: File ObjectId
            
        Returns:
            True if successful
        """
        try:
            album = await Album.get(album_id)
            if not album or album.is_smart:
                return False
            
            if file_id in album.file_ids:
                album.file_ids.remove(file_id)
                album.file_count = len(album.file_ids)
                await album.save()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove file from album: {e}")
            return False
    
    async def get_album_files(
        self,
        album_id: ObjectId,
        limit: int = 100,
        offset: int = 0
    ) -> List[FileRecord]:
        """
        Get files in album.
        
        For smart albums, executes the query.
        For manual albums, retrieves file list.
        
        Args:
            album_id: Album ObjectId
            limit: Max files to return
            offset: Offset for pagination
            
        Returns:
            List of FileRecords
        """
        album = await Album.get(album_id)
        if not album:
            return []
        
        if album.is_smart:
            # Execute smart query
            return await self._execute_smart_query(
                album.smart_query,
                limit,
                offset
            )
        else:
            # Get files from manual list
            file_ids = album.file_ids[offset:offset+limit]
            return await FileRecord.find({"_id": {"$in": file_ids}})
    
    async def _execute_smart_query(
        self,
        query: Dict[str, Any],
        limit: int,
        offset: int
    ) -> List[FileRecord]:
        """
        Execute smart album query.
        
        Args:
            query: MongoDB query
            limit: Max results
            offset: Offset
            
        Returns:
            List of FileRecords
        """
        try:
            # Execute query
            files = await FileRecord.find(query, limit=limit, skip=offset)
            return files
        
        except Exception as e:
            logger.error(f"Failed to execute smart query: {e}")
            return []
    
    async def update_smart_query(
        self,
        album_id: ObjectId,
        query: Dict[str, Any]
    ) -> bool:
        """
        Update smart album query.
        
        Args:
            album_id: Album ObjectId
            query: New query
            
        Returns:
            True if successful
        """
        try:
            album = await Album.get(album_id)
            if not album or not album.is_smart:
                return False
            
            album.smart_query = query
            await album.save()
            
            logger.info(f"Updated smart query for {album.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update smart query: {e}")
            return False
