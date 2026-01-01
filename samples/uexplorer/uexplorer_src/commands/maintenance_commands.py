"""
Maintenance Commands for UExplorer

Encapsulates maintenance operations as standalone commands.
Extracted from MainWindow for modularity.
"""
import asyncio
from typing import TYPE_CHECKING, Callable, List, Optional
from bson import ObjectId
from loguru import logger

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from src.core.locator import ServiceLocator


class MaintenanceUI:
    """
    Provides UI callbacks for maintenance operations.
    
    Allows commands to be decoupled from specific window implementation.
    """
    
    def __init__(
        self,
        parent_widget: "QWidget",
        status_callback: Callable[[str], None],
        progress_callback: Callable[[bool, int, int], None],
        show_error: Callable[[str], None],
        show_warning: Callable[[str], None],
        show_success: Callable[[str], None],
    ):
        """
        Initialize MaintenanceUI.
        
        Args:
            parent_widget: Parent widget for dialogs
            status_callback: Called with status text updates
            progress_callback: Called with (visible, value, max) for progress bar
            show_error: Function to display error dialog
            show_warning: Function to display warning dialog
            show_success: Function to display success dialog
        """
        self.parent = parent_widget
        self.set_status = status_callback
        self.set_progress = progress_callback
        self.show_error = show_error
        self.show_warning = show_warning
        self.show_success = show_success


async def reprocess_selection(
    locator: "ServiceLocator",
    file_ids: List[str],
    ui: MaintenanceUI
) -> None:
    """
    Reprocess selected files through Phase 2/3 pipeline.
    
    Args:
        locator: ServiceLocator for accessing services
        file_ids: List of file ID strings to reprocess
        ui: MaintenanceUI for callbacks
    """
    if not file_ids:
        ui.set_status("No files selected to reprocess")
        return
    
    object_ids = [ObjectId(str(fid)) for fid in file_ids]
    
    try:
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        pipeline = locator.get_system(ProcessingPipeline)
        
        if not pipeline:
            ui.set_status("ProcessingPipeline not available")
            return
        
        # Enqueue Phase 2 (metadata, thumbnails, embeddings)
        task_id = await pipeline.enqueue_phase2(object_ids, force=True)
        
        if task_id:
            ui.set_status(f"Queued {len(object_ids)} files for reprocessing")
            logger.info(f"Reprocess: Queued {len(object_ids)} files - Task {task_id}")
        else:
            ui.set_status("Files already in processing queue")
        
    except Exception as e:
        logger.error(f"Failed to queue reprocessing: {e}")
        ui.set_status(f"Reprocess failed: {e}")


async def reindex_all_files(
    locator: "ServiceLocator",
    ui: MaintenanceUI,
    include_processed: bool = False
) -> None:
    """
    Reindex all files in database via background tasks.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        include_processed: Whether to include already-processed files
    """
    try:
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        pipeline = locator.get_system(ProcessingPipeline)
        
        if not pipeline:
            ui.set_status("ProcessingPipeline not available")
            return
        
        ui.set_status("Starting reindex...")
        
        result = await pipeline.reindex_all(include_processed=include_processed)
        
        total = result.get("total_files", 0)
        batches = result.get("batches_queued", 0)
        
        if total > 0:
            ui.set_status(f"Reindex: {total} files in {batches} batches queued")
            logger.info(f"Reindex started: {total} files, {batches} tasks")
        else:
            ui.set_status("No unprocessed files to reindex")
        
    except Exception as e:
        logger.error(f"Failed to reindex: {e}")
        ui.set_status(f"Reindex failed: {e}")


async def rebuild_all_counts(
    locator: "ServiceLocator",
    ui: MaintenanceUI,
    refresh_panels: Callable[[], None] = None
) -> None:
    """
    Rebuild file counts for tags, albums, directories.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        refresh_panels: Optional callback to refresh UI panels after completion
    """
    from src.ucorefs.services.maintenance_service import MaintenanceService
    
    try:
        maintenance = locator.get_system(MaintenanceService)
        if not maintenance:
            ui.show_error("MaintenanceService not available")
            return
        
        ui.set_progress(True, 0, 100)
        ui.set_status("Rebuilding file counts...")
        
        try:
            result = await maintenance.rebuild_all_counts()
            
            total_updated = (
                result.get("tags_updated", 0) + 
                result.get("albums_updated", 0) + 
                result.get("directories_updated", 0)
            )
            
            message = (
                f"Count rebuild complete!\n\n"
                f"Tags: {result.get('tags_updated', 0)} | "
                f"Albums: {result.get('albums_updated', 0)} | "
                f"Dirs: {result.get('directories_updated', 0)}\n"
                f"Duration: {result.get('duration', 0):.2f}s"
            )
            
            if result.get("errors"):
                message += f"\n\nErrors: {len(result['errors'])}"
            
            # Refresh UI panels
            if refresh_panels:
                refresh_panels()
            
            ui.show_success(message)
            logger.info(f"Count rebuild complete: {total_updated} records updated")
            
        finally:
            ui.set_progress(False, 0, 0)
            ui.set_status("Ready")
            
    except Exception as e:
        ui.show_error(f"Rebuild failed: {e}")
        logger.error(f"Count rebuild failed: {e}")


async def verify_references(
    locator: "ServiceLocator",
    ui: MaintenanceUI
) -> None:
    """
    Verify ObjectId references for data integrity.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
    """
    from src.ucorefs.services.maintenance_service import MaintenanceService
    
    try:
        maintenance = locator.get_system(MaintenanceService)
        if not maintenance:
            ui.show_error("MaintenanceService not available")
            return
        
        ui.set_progress(True, 0, 0)  # Indeterminate
        ui.set_status("Verifying data integrity...")
        
        try:
            result = await maintenance.verify_references()
            
            total_broken = (
                result.get("broken_tag_refs", 0) + 
                result.get("broken_album_refs", 0) + 
                result.get("broken_dir_refs", 0)
            )
            
            if total_broken == 0:
                message = f"✓ All references valid! ({result.get('files_checked', 0)} files checked)"
                ui.show_success(message)
            else:
                message = (
                    f"⚠ Found {total_broken} broken references:\n\n"
                    f"Tags: {result.get('broken_tag_refs', 0)} | "
                    f"Albums: {result.get('broken_album_refs', 0)} | "
                    f"Dirs: {result.get('broken_dir_refs', 0)}\n"
                    f"Files checked: {result.get('files_checked', 0)}\n\n"
                    f"Run 'Cleanup Orphaned Records' to fix."
                )
                ui.show_warning(message)
            
            logger.info(f"Reference verification complete: {total_broken} broken references")
            
        finally:
            ui.set_progress(False, 0, 0)
            ui.set_status("Ready")
            
    except Exception as e:
        ui.show_error(f"Verification failed: {e}")
        logger.error(f"Reference verification failed: {e}")


async def cleanup_orphaned_records(
    locator: "ServiceLocator",
    ui: MaintenanceUI,
    confirmed: bool = False
) -> None:
    """
    Cleanup orphaned references from database.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        confirmed: Whether user has already confirmed the action
    """
    from PySide6.QtWidgets import QMessageBox
    from src.ucorefs.services.maintenance_service import MaintenanceService
    
    try:
        # Confirm action if not already confirmed
        if not confirmed:
            reply = QMessageBox.question(
                ui.parent,
                "Cleanup Orphaned Records",
                "This will remove invalid references from your database.\n\n"
                "This operation is safe but cannot be undone.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        maintenance = locator.get_system(MaintenanceService)
        if not maintenance:
            ui.show_error("MaintenanceService not available")
            return
        
        ui.set_progress(True, 0, 0)  # Indeterminate
        ui.set_status("Cleaning up orphaned records...")
        
        try:
            result = await maintenance.cleanup_orphaned_records()
            
            message = (
                f"✓ Cleanup complete!\n\n"
                f"Files cleaned: {result.get('files_cleaned', 0)}\n"
                f"Tag refs removed: {result.get('tags_removed', 0)}\n"
                f"Album refs removed: {result.get('albums_removed', 0)}"
            )
            
            if result.get("errors"):
                message += f"\n\nErrors: {len(result['errors'])}"
            
            ui.show_success(message)
            logger.info(f"Cleanup complete: {result.get('files_cleaned', 0)} files cleaned")
            
        finally:
            ui.set_progress(False, 0, 0)
            ui.set_status("Ready")
            
    except Exception as e:
        ui.show_error(f"Cleanup failed: {e}")
        logger.error(f"Cleanup failed: {e}")
