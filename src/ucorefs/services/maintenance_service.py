"""
UCoreFS - Maintenance Service

Centralized maintenance operations for UCoreFS data integrity.
"""
from typing import Dict, Any
from loguru import logger
import time

from src.core.base_system import BaseSystem
from src.core.database.manager import DatabaseManager


class MaintenanceService(BaseSystem):
    """
    Centralized maintenance operations for UCoreFS.
    
    Provides methods to:
    - Recalculate all counts across all systems
    - Verify data integrity
    - Clean up orphaned records
    - Background count verification
    """
    
    # Dependency declarations
    depends_on = [DatabaseManager]
    
    async def initialize(self) -> None:
        """Initialize maintenance service."""
        logger.info("MaintenanceService initializing")
        await super().initialize()
        logger.info("MaintenanceService ready")
    
    async def shutdown(self) -> None:
        """Shutdown maintenance service."""
        logger.info("MaintenanceService shutting down")
        await super().shutdown()
    
    async def rebuild_all_counts(self) -> Dict[str, Any]:
        """
        Rebuild all file counts across Tag, Album, and Directory systems.
        
        This method coordinates count recalculation across all three
        organizational systems. Useful for fixing drift after bulk
        operations or database migrations.
        
        Returns:
            Dict with:
                tags_updated: Number of tags with updated counts
                albums_updated: Number of albums with updated counts
                directories_updated: Number of directories with updated counts
                duration: Time taken in seconds
                errors: List of any errors encountered
        
        Example:
            result = await maintenance.rebuild_all_counts()
            logger.info(f"Rebuilt counts in {result['duration']:.2f}s")
        """
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.albums.manager import AlbumManager
        from src.ucorefs.services.fs_service import FSService
        
        start_time = time.time()
        result = {
            "tags_updated": 0,
            "albums_updated": 0,
            "directories_updated": 0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            # Get managers
            tag_manager = self.locator.get_system(TagManager)
            album_manager = self.locator.get_system(AlbumManager)
            fs_service = self.locator.get_system(FSService)
            
            logger.info("Starting full count rebuild...")
            
            # Rebuild tag counts
            if tag_manager:
                try:
                    tag_result = await tag_manager.recalculate_tag_counts()
                    result["tags_updated"] = tag_result.get("updated_count", 0)
                    result["errors"].extend(tag_result.get("errors", []))
                except Exception as e:
                    error_msg = f"Tag count rebuild failed: {e}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Rebuild album counts
            if album_manager:
                try:
                    album_result = await album_manager.recalculate_album_counts()
                    result["albums_updated"] = album_result.get("updated_count", 0)
                    result["errors"].extend(album_result.get("errors", []))
                except Exception as e:
                    error_msg = f"Album count rebuild failed: {e}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Rebuild directory counts
            if fs_service and hasattr(fs_service, 'recalculate_directory_counts'):
                try:
                    dir_result = await fs_service.recalculate_directory_counts()
                    result["directories_updated"] = dir_result.get("directories_updated", 0)
                    result["errors"].extend(dir_result.get("errors", []))
                except Exception as e:
                    error_msg = f"Directory count rebuild failed: {e}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            result["duration"] = time.time() - start_time
            logger.info(f"Count rebuild complete in {result['duration']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Rebuild failed: {e}")
            logger.error(f"Count rebuild failed: {e}")
        
        return result
    
    async def rebuild_album_references(self) -> Dict[str, Any]:
        """
        Rebuild FileRecord.album_ids from Album.file_ids.
        
        This syncs the bidirectional relationship between albums and files.
        Run this after upgrading to bidirectional album tracking.
        
        Returns:
            Dict with files_updated, albums_processed, errors
        """
        from src.ucorefs.models.file_record import FileRecord
        from src.ucorefs.models.album import Album
        
        logger.info("Rebuilding album references...")
        start = time.time()
        
        result = {
            "files_updated": 0,
            "albums_processed": 0,
            "errors": []
        }
        
        try:
            # Step 1: Clear all existing album_ids on files
            logger.info("Clearing existing album_ids...")
            all_files = await FileRecord.find({})
            for file_record in all_files:
                if file_record.album_ids:
                    file_record.album_ids = []
                    await file_record.save()
            
            # Step 2: Rebuild from Album.file_ids
            logger.info("Rebuilding from albums...")
            all_albums = await Album.find({})
            
            for album in all_albums:
                try:
                    if album.is_smart:
                        continue
                    
                    for file_id in album.file_ids:
                        file_record = await FileRecord.get(file_id)
                        if file_record:
                            if album._id not in file_record.album_ids:
                                file_record.album_ids.append(album._id)
                                await file_record.save()
                                result["files_updated"] += 1
                    
                    result["albums_processed"] += 1
                except Exception as e:
                    result["errors"].append(f"Failed album {album._id}: {e}")
                    logger.error(f"Failed album {album._id}: {e}")
            
            duration = time.time() - start
            logger.info(f"Album refs rebuilt in {duration:.2f}s: {result['files_updated']} files")
        
        except Exception as e:
            result["errors"].append(f"Rebuild failed: {e}")
            logger.error(f"Album rebuild failed: {e}")
        
        return result
    
    async def verify_references(self) -> Dict[str, Any]:
        """
        Verify all ObjectId references are valid.
        
        Checks:
        - FileRecord.tag_ids reference existing Tags
        - FileRecord.album_ids reference existing Albums
        - FileRecord.parent_id references existing DirectoryRecord
        
        Returns:
            Dict with:
                broken_tag_refs: Count of broken tag references
                broken_album_refs: Count of broken album references
                broken_dir_refs: Count of broken directory references
                files_checked: Total files checked
        """
        from src.ucorefs.models.file_record import FileRecord
        from src.ucorefs.tags.models import Tag
        from src.ucorefs.albums.models import Album
        from src.ucorefs.models.directory import DirectoryRecord
        
        result = {
            "files_checked": 0,
            "broken_tag_refs": 0,
            "broken_album_refs": 0,
            "broken_dir_refs": 0,
            "errors": []
        }
        
        try:
            logger.info("Verifying ObjectId references...")
            
            # Get all valid IDs
            valid_tag_ids = set([t._id for t in await Tag.find({})])
            valid_album_ids = set([a._id for a in await Album.find({})])
            valid_dir_ids = set([d._id for d in await DirectoryRecord.find({})])
            
            # Check each file
            files = await FileRecord.find({})
            result["files_checked"] = len(files)
            
            for file in files:
                # Check tag references
                if hasattr(file, 'tag_ids') and file.tag_ids:
                    for tag_id in file.tag_ids:
                        if tag_id not in valid_tag_ids:
                            result["broken_tag_refs"] += 1
                            logger.warning(f"File {file._id} references non-existent tag {tag_id}")
                
                # Check album references
                if hasattr(file, 'album_ids') and file.album_ids:
                    for album_id in file.album_ids:
                        if album_id not in valid_album_ids:
                            result["broken_album_refs"] += 1
                            logger.warning(f"File {file._id} references non-existent album {album_id}")
                
                # Check parent directory
                if file.parent_id and file.parent_id not in valid_dir_ids:
                    result["broken_dir_refs"] += 1
                    logger.warning(f"File {file._id} references non-existent parent {file.parent_id}")
            
            logger.info(f"Reference verification complete: {result['broken_tag_refs']} broken tag refs, "
                       f"{result['broken_album_refs']} broken album refs, "
                       f"{result['broken_dir_refs']} broken dir refs")
            
        except Exception as e:
            result["errors"].append(f"Verification failed: {e}")
            logger.error(f"Reference verification failed: {e}")
        
        return result
    
    async def cleanup_orphaned_records(self) -> Dict[str, Any]:
        """
        Remove references to deleted records.
        
        Cleans up:
        - FileRecord.tag_ids pointing to non-existent tags
        - FileRecord.album_ids pointing to non-existent albums
        - Album.file_ids pointing to non-existent files
        
        Returns:
            Dict with:
                files_cleaned: Number of files with references removed
                tags_removed: Total tag references removed
                albums_removed: Total album references removed
        """
        from src.ucorefs.models.file_record import FileRecord
        from src.ucorefs.tags.models import Tag
        from src.ucorefs.albums.models import Album
        
        result = {
            "files_cleaned": 0,
            "tags_removed": 0,
            "albums_removed": 0,
            "errors": []
        }
        
        try:
            logger.info("Cleaning up orphaned records...")
            
            # Get all valid IDs
            valid_tag_ids = set([t._id for t in await Tag.find({})])
            valid_album_ids = set([a._id for a in await Album.find({})])
            valid_file_ids = set([f._id for f in await FileRecord.find({})])
            
            # Clean file records
            files = await FileRecord.find({})
            for file in files:
                cleaned = False
                
                # Clean tag_ids
                if hasattr(file, 'tag_ids') and file.tag_ids:
                    original_count = len(file.tag_ids)
                    file.tag_ids = [tid for tid in file.tag_ids if tid in valid_tag_ids]
                    removed = original_count - len(file.tag_ids)
                    if removed > 0:
                        result["tags_removed"] += removed
                        cleaned = True
                
                # Clean album_ids
                if hasattr(file, 'album_ids') and file.album_ids:
                    original_count = len(file.album_ids)
                    file.album_ids = [aid for aid in file.album_ids if aid in valid_album_ids]
                    removed = original_count - len(file.album_ids)
                    if removed > 0:
                        result["albums_removed"] += removed
                        cleaned = True
                
                if cleaned:
                    await file.save()
                    result["files_cleaned"] += 1
            
            # Clean album file_ids
            albums = await Album.find({})
            for album in albums:
                if not album.is_smart and album.file_ids:
                    original_count = len(album.file_ids)
                    album.file_ids = [fid for fid in album.file_ids if fid in valid_file_ids]
                    if len(album.file_ids) != original_count:
                        album.file_count = len(album.file_ids)
                        await album.save()
            
            logger.info(f"Cleanup complete: {result['files_cleaned']} files cleaned, "
                       f"{result['tags_removed']} tag refs removed, "
                       f"{result['albums_removed']} album refs removed")
            
        except Exception as e:
            result["errors"].append(f"Cleanup failed: {e}")
            logger.error(f"Cleanup failed: {e}")
        
        return result
    
    async def background_count_verification(self) -> None:
        """
        Background task for periodic count verification.
        
        To be called on app idle (every 5 minutes of idle time).
        Silently fixes any count drift detected.
        """
        try:
            logger.debug("Running background count verification...")
            
            # Run full rebuild silently
            result = await self.rebuild_all_counts()
            
            total_updated = (result.get("tags_updated", 0) + 
                           result.get("albums_updated", 0) + 
                           result.get("directories_updated", 0))
            
            if total_updated > 0:
                logger.info(f"Background verification fixed {total_updated} count discrepancies")
            
        except Exception as e:
            logger.error(f"Background count verification failed: {e}")
