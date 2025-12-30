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
    depends_on = ["DatabaseManager", "TaskSystem"]
    
    async def initialize(self) -> None:
        """Initialize maintenance service and register task handlers."""
        logger.info("MaintenanceService initializing")
        
        # Register task handlers with TaskSystem (guaranteed by depends_on)
        from src.core.tasks.system import TaskSystem
        task_system = self.locator.get_system(TaskSystem)
        
        task_system.register_handler('maintenance_background_verification', 
                                    self.background_count_verification)
        task_system.register_handler('maintenance_database_optimization', 
                                    self.database_optimization)
        task_system.register_handler('maintenance_cache_cleanup', 
                                    self.cache_cleanup)
        task_system.register_handler('maintenance_log_rotation', 
                                    self.log_rotation)
        task_system.register_handler('maintenance_orphaned_cleanup', 
                                    self.cleanup_orphaned_file_records)
        task_system.register_handler('maintenance_database_cleanup', 
                                    self.cleanup_old_records)
        
        logger.info("MaintenanceService: Registered 6 task handlers")
        
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
    
    # ==================== New Maintenance Methods ====================
    
    async def database_optimization(self) -> Dict[str, Any]:
        """
        Optimize MongoDB collections and FAISS indexes.
        
        Operations:
        - Compact MongoDB collections (reclaim space)
        - Rebuild MongoDB indexes
        - FAISS index optimization (if needed)
        
        Returns:
            Dict with collections_compacted, indexes_rebuilt, duration, errors
        """
        import asyncio
        
        start_time = time.time()
        result = {
            "collections_compacted": 0,
            "indexes_rebuilt": 0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            logger.info("Starting database optimization...")
            
            # Get database manager
            db_manager = self.locator.get_system(DatabaseManager)
            db = db_manager.db
            
            # Get all collection names
            collection_names = await db.list_collection_names()
            
            # Compact collections (reclaim space)
            for coll_name in collection_names:
                try:
                    # MongoDB compact command (requires admin privileges on some setups)
                    # This may not work on all MongoDB configurations
                    # await db.command({"compact": coll_name})
                    # result["collections_compacted"] += 1
                    
                    # Rebuild indexes using database command (Motor/PyMongo compatible)
                    await db.command({"reIndexCollection": coll_name})
                    result["indexes_rebuilt"] += 1
                    
                    logger.debug(f"Rebuilt indexes for collection: {coll_name}")
                    
                except Exception as e:
                    # Check for CommandNotFound (code 59)
                    if hasattr(e, 'code') and e.code == 59:
                        logger.warning(f"Optimization skipped for {coll_name}: Command not supported (reIndexCollection)")
                    else:
                        error_msg = f"Failed to optimize {coll_name}: {e}"
                        result["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # FAISS index optimization (if exists)
            try:
                # TODO: Add FAISS index optimization when FAISS is integrated
                pass
            except Exception as e:
                logger.debug(f"FAISS optimization skipped: {e}")
            
            result["duration"] = time.time() - start_time
            logger.info(f"Database optimization complete: {result['indexes_rebuilt']} indexes rebuilt in {result['duration']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Database optimization failed: {e}")
            logger.error(f"Database optimization failed: {e}")
        
        return result
    
    async def cache_cleanup(self, max_size_gb: int = 10, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old thumbnails and temporary files.
        
        Args:
            max_size_gb: Maximum cache size in GB
            max_age_days: Delete files older than this many days
        
        Returns:
            Dict with files_deleted, space_freed_gb, duration, errors
        """
        import os
        from pathlib import Path
        from datetime import datetime, timedelta
        
        start_time = time.time()
        result = {
            "files_deleted": 0,
            "space_freed_gb": 0.0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            logger.info(f"Starting cache cleanup (max_size: {max_size_gb}GB, max_age: {max_age_days} days)...")
            
            # Cache directories to clean
            cache_dirs = [
                Path("./data/thumbnails"),
                Path("./data/cache"),
                Path("./data/temp")
            ]
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            total_size_bytes = 0
            
            for cache_dir in cache_dirs:
                if not cache_dir.exists():
                    continue
                
                # Walk through directory
                for file_path in cache_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        # Check file age
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_mtime < cutoff_time:
                            # Delete old file
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            
                            result["files_deleted"] += 1
                            total_size_bytes += file_size
                            
                    except Exception as e:
                        error_msg = f"Failed to delete {file_path}: {e}"
                        result["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            result["space_freed_gb"] = total_size_bytes / (1024**3)  # Convert to GB
            result["duration"] = time.time() - start_time
            
            logger.info(f"Cache cleanup complete: {result['files_deleted']} files deleted, "
                       f"{result['space_freed_gb']:.2f}GB freed in {result['duration']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Cache cleanup failed: {e}")
            logger.error(f"Cache cleanup failed: {e}")
        
        return result
    
    async def log_rotation(self, max_log_files: int = 10, max_log_size_mb: int = 100) -> Dict[str, Any]:
        """
        Rotate and archive log files.
        
        Args:
            max_log_files: Maximum number of log files to keep
            max_log_size_mb: Archive logs larger than this size
        
        Returns:
            Dict with files_rotated, files_deleted, duration, errors
        """
        import os
        from pathlib import Path
        import shutil
        
        start_time = time.time()
        result = {
            "files_rotated": 0,
            "files_deleted": 0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            logger.info(f"Starting log rotation (max_files: {max_log_files}, max_size: {max_log_size_mb}MB)...")
            
            log_dir = Path("./logs")
            if not log_dir.exists():
                logger.info("Log directory does not exist, skipping rotation")
                return result
            
            # Get all log files sorted by modification time (oldest first)
            log_files = sorted(
                [f for f in log_dir.glob("*.log")],
                key=lambda x: x.stat().st_mtime
            )
            
            # Delete excess log files (keep only max_log_files most recent)
            if len(log_files) > max_log_files:
                files_to_delete = log_files[:len(log_files) - max_log_files]
                for log_file in files_to_delete:
                    try:
                        log_file.unlink()
                        result["files_deleted"] += 1
                        logger.debug(f"Deleted old log: {log_file.name}")
                    except Exception as e:
                        error_msg = f"Failed to delete {log_file}: {e}"
                        result["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Rotate large log files
            max_size_bytes = max_log_size_mb * 1024 * 1024
            for log_file in log_dir.glob("*.log"):
                try:
                    if log_file.stat().st_size > max_size_bytes:
                        # Archive large log
                        archive_name = log_file.with_suffix(f".{int(time.time())}.log.old")
                        shutil.move(str(log_file), str(archive_name))
                        result["files_rotated"] += 1
                        logger.debug(f"Rotated large log: {log_file.name} -> {archive_name.name}")
                except Exception as e:
                    error_msg = f"Failed to rotate {log_file}: {e}"
                    result["errors"].append(error_msg)
                    logger.warning(error_msg)
            
            result["duration"] = time.time() - start_time
            logger.info(f"Log rotation complete: {result['files_rotated']} rotated, "
                       f"{result['files_deleted']} deleted in {result['duration']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Log rotation failed: {e}")
            logger.error(f"Log rotation failed: {e}")
        
        return result
    
    async def cleanup_orphaned_file_records(self) -> int:
        """
        Remove FileRecords for files that no longer exist on disk.
        
        Returns:
            Number of records removed
        """
        import os
        from src.ucorefs.models.file_record import FileRecord
        
        try:
            logger.info("Starting orphaned file records cleanup...")
            
            removed_count = 0
            
            # Get all file records
            files = await FileRecord.find({})
            
            for file in files:
                # Check if file exists on disk
                if not os.path.exists(file.path):
                    # File doesn't exist, remove record
                    await file.delete()
                    removed_count += 1
                    logger.debug(f"Removed orphaned record: {file.path}")
            
            logger.info(f"Orphaned file cleanup complete: {removed_count} records removed")
            return removed_count
            
        except Exception as e:
            logger.error(f"Orphaned file cleanup failed: {e}")
            return 0
    
    async def cleanup_old_records(
        self,
        task_retention_days: int = 30,
        journal_retention_days: int = 90
    ) -> Dict[str, Any]:
        """
        Clean up old TaskRecord and JournalEntry documents.
        
        Args:
            task_retention_days: Keep TaskRecords newer than this (default: 30 days)
            journal_retention_days: Keep JournalEntries newer than this (default: 90 days)
        
        Returns:
            Dict with tasks_deleted, journal_deleted, duration, errors
        """
        from datetime import datetime, timedelta
        
        start_time = time.time()
        result = {
            "tasks_deleted": 0,
            "journal_deleted": 0,
            "duration": 0.0,
            "errors": []
        }
        
        try:
            logger.info(f"Starting database cleanup (tasks: {task_retention_days}d, journal: {journal_retention_days}d)...")
            
            # Calculate cutoff timestamps
            task_cutoff = int((datetime.utcnow() - timedelta(days=task_retention_days)).timestamp())
            journal_cutoff = int((datetime.utcnow() - timedelta(days=journal_retention_days)).timestamp())
            
            # Clean up old TaskRecords (completed or failed only, keep pending/running)
            try:
                from src.core.tasks.models import TaskRecord
                
                old_tasks = await TaskRecord.find({
                    "status": {"$in": ["completed", "failed"]},
                    "created_at": {"$lt": task_cutoff}
                })
                
                for task in old_tasks:
                    await task.delete()
                    result["tasks_deleted"] += 1
                
                logger.debug(f"Deleted {result['tasks_deleted']} old task records")
                
            except (ImportError, Exception) as e:
                logger.debug(f"TaskRecord cleanup skipped: {e}")
            
            # Clean up old JournalEntries
            try:
                from src.core.journal.models import JournalEntry
                
                old_journal = await JournalEntry.find({
                    "timestamp": {"$lt": journal_cutoff}
                })
                
                for entry in old_journal:
                    await entry.delete()
                    result["journal_deleted"] += 1
                
                logger.debug(f"Deleted {result['journal_deleted']} old journal entries")
                
            except (ImportError, Exception) as e:
                logger.debug(f"JournalEntry cleanup skipped: {e}")
            
            result["duration"] = time.time() - start_time
            logger.info(f"Database cleanup complete: {result['tasks_deleted']} tasks, "
                       f"{result['journal_deleted']} journal entries removed in {result['duration']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Database cleanup failed: {e}")
            logger.error(f"Database cleanup failed: {e}")
        
        return result

