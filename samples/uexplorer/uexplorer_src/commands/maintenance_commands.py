"""
Maintenance Commands for UExplorer

Simplified commands using TaskSystem.submit_background for non-blocking execution.
Results are delivered via TaskSystem signals connected in MainWindow.
"""
import asyncio
from typing import TYPE_CHECKING, Callable, List
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


def _get_task_system(locator: "ServiceLocator"):
    """Get TaskSystem from locator, or None if not available."""
    try:
        from src.core.tasks.system import TaskSystem
        return locator.get_system(TaskSystem)
    except (KeyError, ImportError):
        return None


async def reprocess_selection(
    locator: "ServiceLocator",
    file_ids: List[str],
    ui: MaintenanceUI
) -> None:
    """
    Reprocess selected files through Phase 2/3 pipeline.
    
    Uses TaskSystem.submit_background for non-blocking execution.
    Results delivered via TaskSystem signals.
    
    Args:
        locator: ServiceLocator for accessing services
        file_ids: List of file ID strings to reprocess
        ui: MaintenanceUI for callbacks
    """
    if not file_ids:
        ui.set_status("No files selected to reprocess")
        return
    
    task_system = _get_task_system(locator)
    if not task_system:
        ui.set_status("TaskSystem not available")
        return
    
    try:
        # Submit via non-blocking background API
        task_id = task_system.submit_background(
            "uexplorer.reprocess",
            file_ids,
            priority=task_system.PRIORITY_HIGH
        )
        
        ui.set_status(f"Queued {len(file_ids)} files for reprocessing")
        logger.info(f"Reprocess submitted: {len(file_ids)} files, task {task_id[:8]}")
        
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
    
    Uses TaskSystem.submit_background for non-blocking execution.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        include_processed: Whether to include already-processed files
    """
    task_system = _get_task_system(locator)
    if not task_system:
        ui.set_status("TaskSystem not available")
        return
    
    try:
        ui.set_status("Starting reindex...")
        
        task_id = task_system.submit_background(
            "uexplorer.reindex",
            include_processed,
            priority=task_system.PRIORITY_NORMAL
        )
        
        ui.set_status(f"Reindex started (task: {task_id[:8]}...)")
        logger.info(f"Reindex submitted, task {task_id[:8]}")
        
    except Exception as e:
        logger.error(f"Failed to start reindex: {e}")
        ui.set_status(f"Reindex failed: {e}")


async def rebuild_all_counts(
    locator: "ServiceLocator",
    ui: MaintenanceUI,
    refresh_panels: Callable[[], None] = None
) -> None:
    """
    Rebuild file counts for tags, albums, directories.
    
    Uses TaskSystem.submit_background for non-blocking execution.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        refresh_panels: Optional callback to refresh UI panels after completion
    """
    task_system = _get_task_system(locator)
    if not task_system:
        ui.set_status("TaskSystem not available")
        return
    
    try:
        ui.set_progress(True, 0, 100)
        ui.set_status("Rebuilding file counts...")
        
        task_id = task_system.submit_background(
            "uexplorer.rebuild_counts",
            priority=task_system.PRIORITY_NORMAL
        )
        
        ui.set_status(f"Count rebuild started (task: {task_id[:8]}...)")
        logger.info(f"Rebuild counts submitted, task {task_id[:8]}")
        
        # Note: refresh_panels will be called via signal handler when task completes
        # For now, schedule it with delay as workaround
        if refresh_panels:
            async def _delayed_refresh():
                await asyncio.sleep(2)  # Give task time to complete
                refresh_panels()
            asyncio.create_task(_delayed_refresh())
        
    except Exception as e:
        logger.error(f"Failed to start count rebuild: {e}")
        ui.set_status(f"Rebuild failed: {e}")
        ui.set_progress(False, 0, 0)


async def verify_references(
    locator: "ServiceLocator",
    ui: MaintenanceUI
) -> None:
    """
    Verify ObjectId references for data integrity.
    
    Uses TaskSystem.submit_background for non-blocking execution.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
    """
    task_system = _get_task_system(locator)
    if not task_system:
        ui.set_status("TaskSystem not available")
        return
    
    try:
        ui.set_progress(True, 0, 0)  # Indeterminate
        ui.set_status("Verifying data integrity...")
        
        task_id = task_system.submit_background(
            "uexplorer.verify_refs",
            priority=task_system.PRIORITY_NORMAL
        )
        
        ui.set_status(f"Verification started (task: {task_id[:8]}...)")
        logger.info(f"Verify refs submitted, task {task_id[:8]}")
        
    except Exception as e:
        logger.error(f"Failed to start verification: {e}")
        ui.set_status(f"Verification failed: {e}")
        ui.set_progress(False, 0, 0)


async def cleanup_orphaned_records(
    locator: "ServiceLocator",
    ui: MaintenanceUI,
    confirmed: bool = False
) -> None:
    """
    Cleanup orphaned references from database.
    
    Uses TaskSystem.submit_background for non-blocking execution.
    
    Args:
        locator: ServiceLocator for accessing services
        ui: MaintenanceUI for callbacks
        confirmed: Whether user has already confirmed the action
    """
    from PySide6.QtWidgets import QMessageBox
    
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
    
    task_system = _get_task_system(locator)
    if not task_system:
        ui.set_status("TaskSystem not available")
        return
    
    try:
        ui.set_progress(True, 0, 0)  # Indeterminate
        ui.set_status("Cleaning up orphaned records...")
        
        task_id = task_system.submit_background(
            "uexplorer.cleanup",
            priority=task_system.PRIORITY_NORMAL
        )
        
        ui.set_status(f"Cleanup started (task: {task_id[:8]}...)")
        logger.info(f"Cleanup submitted, task {task_id[:8]}")
        
    except Exception as e:
        logger.error(f"Failed to start cleanup: {e}")
        ui.set_status(f"Cleanup failed: {e}")
        ui.set_progress(False, 0, 0)
