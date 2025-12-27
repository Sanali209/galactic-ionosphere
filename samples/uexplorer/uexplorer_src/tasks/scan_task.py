"""
Scan Directory Task

Background task for scanning library directories and updating the database.
Demonstrates Foundation's TaskSystem with progress reporting.
"""
import asyncio
from typing import Dict, Any
from pathlib import Path
from loguru import logger

# Note: Task base class would be imported from Foundation's TaskSystem
# For now, we'll use the existing DiscoveryService pattern which already
# supports background tasks via the TaskSystem


async def scan_directory_task(locator, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task to scan a directory.
    
    This demonstrates how to create a background task that:
    1. Reports progress updates
    2. Can be monitored from the UI
    3. Integrates with Foundation's TaskSystem
    
    Args:
        locator: ServiceLocator instance
        params: Task parameters
            - path: str, directory path to scan
            - recursive: bool, scan subdirectories (default: True)
    
    Returns:
        Dict with scan results:
            - files_added: int
            - dirs_added: int
            - files_modified: int
            - files_deleted: int
            - dirs_deleted: int
    """
    from src.ucorefs.discovery.service import DiscoveryService
    
    path = params.get('path')
    recursive = params.get('recursive', True)
    
    logger.info(f"Starting background scan of {path}")
    
    # Get discovery service
    discovery = locator.get_system(DiscoveryService)
    
    # The DiscoveryService already integrates with TaskSystem
    # via scan_root() which submits tasks internally
    # This demonstrates the pattern
    
    if not discovery:
        logger.error("DiscoveryService not available")
        return {
            'error': 'DiscoveryService not available',
            'files_added': 0,
            'dirs_added': 0,
            'files_modified': 0,
            'files_deleted': 0,
            'dirs_deleted': 0
        }
    
    # Perform the scan
    # Note: The actual implementation uses TaskSystem internally
    result = await discovery.scan_directory(path, recursive=recursive)
    
    logger.info(f"Scan complete for {path}: {result}")
    
    return result


# Example of custom task class (if we were to create new task types)
"""
from src.core.tasks.system import Task

class ScanDirectoryTask(Task):
    '''Custom scan task with progress reporting.'''
    
    async def execute(self, params: dict):
        '''Execute the scan task.'''
        path = params['path']
        
        # Report initial progress
        await self.report_progress(0, f"Starting scan of {path}")
        
        # Get services from locator
        discovery = self.locator.get_system(DiscoveryService)
        
        # Progress callback
        async def on_progress(current: int, total: int, item: str):
            percent = int((current / total) * 100) if total > 0 else 0
            await self.report_progress(
                percent,
                f"Scanned {current}/{total} items: {item}"
            )
        
        # Scan with progress
        result = await discovery.scan_directory_with_progress(
            path,
            progress_callback=on_progress
        )
        
        # Report completion
        await self.report_progress(100, "Scan complete")
        
        return result
"""
