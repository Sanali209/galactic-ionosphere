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
    
    depends_on = ["DatabaseManager"]
    
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
        
        Updates both Album.file_ids and FileRecord.album_ids (bidirectional).
        
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
                # Update album
                album.file_ids.append(file_id)
                album.file_count = len(album.file_ids)
                await album.save()
                
                # Update file record (bidirectional relationship)
                from src.ucorefs.models.file_record import FileRecord
                file_record = await FileRecord.get(file_id)
                if file_record and album_id not in file_record.album_ids:
                    file_record.album_ids.append(album_id)
                    await file_record.save()
                
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
        
        Updates both Album.file_ids and FileRecord.album_ids (bidirectional).
        
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
                # Update album
                album.file_ids.remove(file_id)
                album.file_count = len(album.file_ids)
                await album.save()
                
                # Update file record (bidirectional relationship)
                from src.ucorefs.models.file_record import FileRecord
                file_record = await FileRecord.get(file_id)
                if file_record and album_id in file_record.album_ids:
                    file_record.album_ids.remove(album_id)
                    await file_record.save()
            
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
    
    # ==================== Count Management ====================
    
    async def recalculate_album_counts(self) -> dict:
        """
        Recalculate file_count for all albums.
        
        For manual albums: count = len(file_ids)
        For smart albums: execute query and count results
        
        Returns:
            Dict with:
                total_albums: Total number of albums processed
                updated_count: Number of albums with changed counts
                errors: List of error messages
        """
        result = {
            "total_albums": 0,
            "updated_count": 0,
            "errors": []
        }
        
        try:
            # Get all albums
            all_albums = await Album.find({})
            result["total_albums"] = len(all_albums)
            
            logger.info(f"Recalculating counts for {result['total_albums']} albums...")
            
            # Process each album
            for album in all_albums:
                try:
                    if album.is_smart:
                        # Count via smart query
                        files = await FileRecord.find(album.smart_query)
                        count = len(files)
                    else:
                        # Count manual album file list
                        count = len(album.file_ids) if album.file_ids else 0
                    
                    # Update if different
                    if count != album.file_count:
                        album.file_count = count
                        await album.save()
                        result["updated_count"] += 1
                        logger.debug(f"Updated album '{album.name}': {count} files")
                
                except Exception as e:
                    error_msg = f"Failed to update album {album._id}: {e}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Album count recalculation complete: {result['updated_count']} updated")
            
        except Exception as e:
            error_msg = f"Album count recalculation failed: {e}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    async def update_album_count(self, album_id: ObjectId) -> int:
        """
        Update file_count for a specific album.
        
        Args:
            album_id: Album ObjectId
            
        Returns:
            New count value
            
        Raises:
            ValueError: If album not found
        """
        album = await Album.get(album_id)
        if not album:
            raise ValueError(f"Album not found: {album_id}")
        
        # Calculate count
        if album.is_smart:
            files = await FileRecord.find(album.smart_query)
            count = len(files)
        else:
            count = len(album.file_ids) if album.file_ids else 0
        
        # Update album
        album.file_count = count
        await album.save()
        
        logger.debug(f"Updated album '{album.name}' count: {count}")
        return count
    
    async def get_album_count(self, album_id: ObjectId) -> int:
        """
        Get current file count for an album without updating database.
        
        For smart albums, executes query to get real-time count.
        For manual albums, returns cached count.
        
        Args:
            album_id: Album ObjectId
            
        Returns:
            File count
        """
        album = await Album.get(album_id)
        if not album:
            return 0
        
        if album.is_smart:
            # Real-time count for smart albums
            try:
                files = await FileRecord.find(album.smart_query)
                return len(files)
            except Exception as e:
                logger.error(f"Failed to count smart album files: {e}")
                return 0
        else:
            # Cached count for manual albums
            return album.file_count

