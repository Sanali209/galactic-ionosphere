"""
UCoreFS - Discovery Service

Main service orchestrating filesystem discovery and synchronization.
"""
from typing import Optional
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.tasks.system import TaskSystem
from src.ucorefs.core.fs_service import FSService
from src.ucorefs.discovery.library_manager import LibraryManager
from src.ucorefs.discovery.scanner import DirectoryScanner
from src.ucorefs.discovery.diff import DiffDetector
from src.ucorefs.discovery.sync import SyncManager


class DiscoveryService(BaseSystem):
    """
    Orchestrates filesystem discovery and database synchronization.
    
    Coordinates the scanning, diff detection, and database sync processes.
    Integrates with TaskSystem for background scanning.
    """
    
    async def initialize(self) -> None:
        """Initialize discovery service and dependencies."""
        logger.info("DiscoveryService initializing")
        
        # Get dependencies
        self.fs_service = self.locator.get_system(FSService)
        self.task_system = self.locator.get_system(TaskSystem)
        
        # Get ProcessingPipeline (optional - may not be registered yet)
        self.processing_pipeline = None
        try:
            from src.ucorefs.processing.pipeline import ProcessingPipeline
            self.processing_pipeline = self.locator.get_system(ProcessingPipeline)
        except (KeyError, ImportError):
            logger.info("ProcessingPipeline not available - Phase 2 auto-queue disabled")
        
        # Initialize components
        self.library_manager = LibraryManager(self.fs_service)
        self.scanner = DirectoryScanner(self.library_manager, batch_size=200)  # Phase 1 batch size
        self.diff_detector = DiffDetector()
        self.sync_manager = SyncManager(self.fs_service)
        
        # Register task handlers
        self.task_system.register_handler("scan_library_root", self._handle_scan_root)
        
        await super().initialize()
        logger.info("DiscoveryService ready")
    
    async def shutdown(self) -> None:
        """Shutdown discovery service."""
        logger.info("DiscoveryService shutting down")
        await super().shutdown()
    
    async def scan_root(
        self,
        root_id: ObjectId,
        background: bool = True
    ) -> Optional[str]:
        """
        Scan a library root for changes.
        
        Args:
            root_id: Library root ObjectId to scan
            background: If True, submit as background task
            
        Returns:
            Task ID if background, None if immediate
        """
        if background:
            # Submit to task system
            task_id = await self.task_system.submit(
                "scan_library_root",
                f"Scan library root",
                str(root_id)
            )
            logger.info(f"Submitted scan task: {task_id}")
            return task_id
        else:
            # Run immediately
            await self._scan_root_impl(root_id)
            return None
    
    async def scan_all_roots(self, background: bool = True) -> list:
        """
        Scan all enabled library roots.
        
        Args:
            background: If True, submit as background tasks
            
        Returns:
            List of task IDs if background
        """
        roots = await self.library_manager.get_enabled_roots()
        task_ids = []
        
        for root in roots:
            task_id = await self.scan_root(root._id, background=background)
            if task_id:
                task_ids.append(task_id)
        
        return task_ids
    
    async def _scan_root_impl(self, root_id: ObjectId) -> dict:
        """
        Implementation of root scanning.
        
        Args:
            root_id: Root directory ObjectId
            
        Returns:
            Statistics dict
        """
        # Get root record
        from src.ucorefs.models.directory import DirectoryRecord
        root = await DirectoryRecord.get(root_id)
        
        if not root or not root.is_root:
            raise ValueError(f"Invalid library root: {root_id}")
        
        logger.info(f"Scanning root: {root.path}")
        
        total_stats = {
            "files_added": 0,
            "dirs_added": 0,
            "files_modified": 0,
            "files_deleted": 0,
            "dirs_deleted": 0
        }
        
        # Track all visited paths for deletion detection
        visited_paths = set()
        
        # Scan in batches
        for batch in self.scanner.scan_directory(
            root.path,
            root.watch_extensions,
            root.blacklist_paths,
            recursive=True
        ):
            # Accumulate paths
            for item in batch:
                visited_paths.add(item.path)
                
            # Detect changes (INCREMENTAL: Only Adds/Mods)
            diff = await self.diff_detector.detect_changes(
                batch, 
                root.path, 
                incremental=True
            )
            
            # Apply changes
            if diff.total_changes > 0:
                stats = await self.sync_manager.apply_changes(diff, str(root._id))
                
                # Accumulate stats
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
                
                # Queue Phase 2 processing for new files
                added_ids = stats.get("added_file_ids", [])
                if added_ids and self.processing_pipeline:
                    await self.processing_pipeline.enqueue_phase2(added_ids)
                    
                # Publish event for real-time UI updates
                try:
                    from src.core.commands.bus import CommandBus
                    bus = self.locator.get_system(CommandBus)
                    if hasattr(bus, 'publish'):
                        await bus.publish("filesystem.updated", {
                            "root_id": str(root._id),
                            "stats": stats
                        })
                except (KeyError, ImportError, AttributeError):
                    pass
        
        # Final pass: Detect deletions (compare all DB files vs visited_paths)
        del_diff = await self.diff_detector.detect_deletions(visited_paths, root.path)
        
        if del_diff.total_changes > 0:
             del_stats = await self.sync_manager.apply_changes(del_diff, str(root._id))
             
             # Accumulate stats
             for key in total_stats:
                total_stats[key] += del_stats.get(key, 0)
             
             # Publish deletions event
             try:
                 from src.core.commands.bus import CommandBus
                 bus = self.locator.get_system(CommandBus)
                 if hasattr(bus, 'publish'):
                     await bus.publish("filesystem.updated", {
                        "root_id": str(root._id),
                        "stats": del_stats
                     })
             except (KeyError, ImportError, AttributeError):
                 pass

        logger.info(f"Scan complete for {root.path}: {total_stats}")
        return total_stats
    
    async def _handle_scan_root(self, root_id_str: str):
        """
        Handle scan_library_root task.
        
        Args:
            root_id_str: Root ObjectId string
            
        Returns:
            Review of scan results
        """
        root_id = ObjectId(root_id_str)
        return await self._scan_root_impl(root_id)
