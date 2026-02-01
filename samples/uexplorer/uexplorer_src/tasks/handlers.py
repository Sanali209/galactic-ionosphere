"""
UExplorer Task Handlers

Background task handlers registered with TaskSystem.
These run in BackgroundTaskRunner's QThread for non-blocking UI.

Usage:
    from uexplorer_src.tasks.handlers import register_handlers
    register_handlers(task_system)
"""
from typing import List, Dict, Any
from bson import ObjectId
from loguru import logger


async def handle_reprocess_files(file_ids: List[str]) -> Dict[str, Any]:
    """
    Reprocess files through Phase 2/3 pipeline.
    
    Handler: uexplorer.reprocess
    
    Args:
        file_ids: List of file ID strings to reprocess
        
    Returns:
        Dict with task_id and file_count
    """
    from src.ucorefs.processing.pipeline import ProcessingPipeline
    from src.core.locator import get_active_locator
    
    try:
        locator = get_active_locator()
        pipeline = locator.get_system(ProcessingPipeline)
        
        oids = [ObjectId(fid) for fid in file_ids]
        task_id = await pipeline.enqueue_phase2(oids, force=True)
        
        logger.info(f"[uexplorer.reprocess] Queued {len(oids)} files, task: {task_id}")
        return {"task_id": task_id, "file_count": len(oids), "success": True}
        
    except Exception as e:
        logger.error(f"[uexplorer.reprocess] Failed: {e}")
        return {"success": False, "error": str(e)}


async def handle_reindex_all(include_processed: bool = False) -> Dict[str, Any]:
    """
    Reindex all files in database.
    
    Handler: uexplorer.reindex
    
    Args:
        include_processed: Whether to include already-processed files
        
    Returns:
        Dict with total_files and batches_queued
    """
    from src.ucorefs.processing.pipeline import ProcessingPipeline
    from src.core.locator import get_active_locator
    
    try:
        locator = get_active_locator()
        pipeline = locator.get_system(ProcessingPipeline)
        
        result = await pipeline.reindex_all(include_processed=include_processed)
        
        logger.info(f"[uexplorer.reindex] {result.get('total_files', 0)} files queued")
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"[uexplorer.reindex] Failed: {e}")
        return {"success": False, "error": str(e)}


async def handle_rebuild_counts() -> Dict[str, Any]:
    """
    Rebuild file counts for tags, albums, directories.
    
    Handler: uexplorer.rebuild_counts
    
    Returns:
        Dict with tags_updated, albums_updated, directories_updated
    """
    from src.ucorefs.services.maintenance_service import MaintenanceService
    from src.core.locator import get_active_locator
    
    try:
        locator = get_active_locator()
        maintenance = locator.get_system(MaintenanceService)
        
        result = await maintenance.rebuild_all_counts()
        
        total = (
            result.get("tags_updated", 0) + 
            result.get("albums_updated", 0) + 
            result.get("directories_updated", 0)
        )
        logger.info(f"[uexplorer.rebuild_counts] Updated {total} records")
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"[uexplorer.rebuild_counts] Failed: {e}")
        return {"success": False, "error": str(e)}


async def handle_verify_references() -> Dict[str, Any]:
    """
    Verify ObjectId references for data integrity.
    
    Handler: uexplorer.verify_refs
    
    Returns:
        Dict with files_checked, broken_tag_refs, broken_album_refs, broken_dir_refs
    """
    from src.ucorefs.services.maintenance_service import MaintenanceService
    from src.core.locator import get_active_locator
    
    try:
        locator = get_active_locator()
        maintenance = locator.get_system(MaintenanceService)
        
        result = await maintenance.verify_references()
        
        broken = (
            result.get("broken_tag_refs", 0) + 
            result.get("broken_album_refs", 0) + 
            result.get("broken_dir_refs", 0)
        )
        logger.info(f"[uexplorer.verify_refs] Checked {result.get('files_checked', 0)} files, {broken} broken")
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"[uexplorer.verify_refs] Failed: {e}")
        return {"success": False, "error": str(e)}


async def handle_cleanup_orphans() -> Dict[str, Any]:
    """
    Cleanup orphaned references from database.
    
    Handler: uexplorer.cleanup
    
    Returns:
        Dict with files_cleaned, tags_removed, albums_removed
    """
    from src.ucorefs.services.maintenance_service import MaintenanceService
    from src.core.locator import get_active_locator
    
    try:
        locator = get_active_locator()
        maintenance = locator.get_system(MaintenanceService)
        
        result = await maintenance.cleanup_orphaned_records()
        
        logger.info(f"[uexplorer.cleanup] Cleaned {result.get('files_cleaned', 0)} files")
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"[uexplorer.cleanup] Failed: {e}")
        return {"success": False, "error": str(e)}


def register_handlers(task_system) -> int:
    """
    Register all UExplorer handlers with TaskSystem.
    
    Call this after TaskSystem is initialized in main.py.
    
    Args:
        task_system: TaskSystem instance
        
    Returns:
        Number of handlers registered
    """
    handlers = [
        ("uexplorer.reprocess", handle_reprocess_files),
        ("uexplorer.reindex", handle_reindex_all),
        ("uexplorer.rebuild_counts", handle_rebuild_counts),
        ("uexplorer.verify_refs", handle_verify_references),
        ("uexplorer.cleanup", handle_cleanup_orphans),
    ]
    
    logger.info(f"UExplorer register_handlers: TaskSystem id={id(task_system)}")
    logger.info(f"UExplorer register_handlers: Before registration, handlers={list(task_system._handlers.keys())}")
    
    for name, handler in handlers:
        task_system.register_handler(name, handler)
        logger.debug(f"Registered handler: {name}")
    
    logger.info(f"UExplorer register_handlers: After registration, handlers={list(task_system._handlers.keys())}")
    logger.info(f"UExplorer: Registered {len(handlers)} task handlers")
    return len(handlers)

