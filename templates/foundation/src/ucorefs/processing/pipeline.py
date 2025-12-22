"""
UCoreFS - Processing Pipeline

Background processing service integrated with TaskSystem.
Manages Phase 2/3 enrichment queues and worker tasks.
"""
from typing import List, Optional, Set
from bson import ObjectId
from loguru import logger
import asyncio

from src.core.base_system import BaseSystem
from src.core.tasks.system import TaskSystem
from src.ucorefs.models.base import ProcessingState
from src.ucorefs.models.file_record import FileRecord


class ProcessingPipeline(BaseSystem):
    """
    Queued pipeline for Phase 2/3 file processing.
    
    Integrates with TaskSystem for background task execution.
    
    Batch Sizes:
    - Phase 2: 20 items (thumbnails, metadata, embeddings)
    - Phase 3: 1 item (AI detection, descriptions)
    """
    
    PHASE2_BATCH_SIZE = 20
    PHASE3_BATCH_SIZE = 1
    
    async def initialize(self) -> None:
        """Initialize pipeline and register task handlers."""
        logger.info("ProcessingPipeline initializing")
        
        # Get dependencies
        self.task_system = self.locator.get_system(TaskSystem)
        
        # Register task handlers
        self.task_system.register_handler("process_phase2_batch", self._handle_phase2_batch)
        self.task_system.register_handler("process_phase3_item", self._handle_phase3_item)
        
        # Track pending work to avoid duplicates
        self._phase2_pending: Set[str] = set()
        self._phase3_pending: Set[str] = set()
        
        # Subscribe to ChangeMonitor for auto-processing
        try:
            from src.core.commands.bus import CommandBus
            bus = self.locator.get_system(CommandBus)
            if hasattr(bus, 'subscribe'):
                bus.subscribe("file.created", self._on_file_created)
                bus.subscribe("file.modified", self._on_file_modified)
                logger.info("ProcessingPipeline subscribed to file events")
        except (KeyError, ImportError):
            logger.debug("CommandBus not available for file events")
        
        await super().initialize()
        logger.info("ProcessingPipeline ready")
    
    async def shutdown(self) -> None:
        """Shutdown pipeline."""
        logger.info("ProcessingPipeline shutting down")
        await super().shutdown()
    
    # ==================== Enqueue Methods ====================
    
    async def enqueue_phase2(self, file_ids: List[ObjectId]) -> Optional[str]:
        """
        Enqueue files for Phase 2 processing.
        
        Args:
            file_ids: List of file ObjectIds to process
            
        Returns:
            Task ID if submitted, None if already pending
        """
        # Filter out already pending
        new_ids = [fid for fid in file_ids if str(fid) not in self._phase2_pending]
        
        if not new_ids:
            return None
        
        # Mark as pending
        for fid in new_ids:
            self._phase2_pending.add(str(fid))
        
        # Submit task
        task_id = await self.task_system.submit(
            "process_phase2_batch",
            f"Phase 2: Process {len(new_ids)} files",
            ",".join(str(fid) for fid in new_ids)
        )
        
        logger.info(f"Enqueued Phase 2 task {task_id} for {len(new_ids)} files")
        return task_id
    
    async def enqueue_phase3(self, file_id: ObjectId) -> Optional[str]:
        """
        Enqueue single file for Phase 3 processing.
        
        Args:
            file_id: File ObjectId to process
            
        Returns:
            Task ID if submitted, None if already pending
        """
        file_id_str = str(file_id)
        
        if file_id_str in self._phase3_pending:
            return None
        
        self._phase3_pending.add(file_id_str)
        
        task_id = await self.task_system.submit(
            "process_phase3_item",
            f"Phase 3: Process file {file_id_str[:8]}...",
            file_id_str
        )
        
        logger.info(f"Enqueued Phase 3 task {task_id} for {file_id_str}")
        return task_id
    
    # ==================== Task Handlers ====================
    
    async def _handle_phase2_batch(self, file_ids_str: str) -> dict:
        """
        Handle Phase 2 batch processing task.
        
        Operations run via ExtractorRegistry:
        1. Generate thumbnails
        2. Extract metadata (EXIF, XMP)
        3. Compute basic embeddings
        """
        from src.ucorefs.extractors import ExtractorRegistry
        
        file_ids = [ObjectId(fid) for fid in file_ids_str.split(",") if fid]
        
        results = {
            "processed": 0,
            "errors": 0,
            "by_extractor": {}
        }
        
        # Load files
        files = []
        for file_id in file_ids:
            try:
                file = await FileRecord.get(file_id)
                if file:
                    files.append(file)
            except Exception as e:
                logger.error(f"Failed to load file {file_id}: {e}")
                results["errors"] += 1
        
        if not files:
            return results
        
        # Get Phase 2 extractors
        extractors = ExtractorRegistry.get_for_phase(2, locator=self.locator)
        
        for extractor in extractors:
            try:
                # Filter to files this extractor can process
                processable = [f for f in files if extractor.can_process(f)]
                
                if processable:
                    extractor_results = await extractor.process(processable)
                    success_count = sum(1 for v in extractor_results.values() if v)
                    results["by_extractor"][extractor.name] = success_count
                    results["processed"] += success_count
                    
            except Exception as e:
                logger.error(f"Extractor {extractor.name} failed: {e}")
                results["errors"] += 1
        
        # Remove from pending
        for file_id in file_ids:
            self._phase2_pending.discard(str(file_id))
        
        # Publish event for UI updates
        await self._publish_progress("phase2.complete", results)
        
        return results
    
    async def _handle_phase3_item(self, file_id_str: str) -> dict:
        """
        Handle Phase 3 single item processing task.
        
        Operations run via ExtractorRegistry:
        1. BLIP captioning
        2. GroundingDINO object detection
        3. Other Phase 3 extractors
        """
        from src.ucorefs.extractors import ExtractorRegistry
        
        file_id = ObjectId(file_id_str)
        
        results = {
            "processed": False,
            "by_extractor": {},
            "errors": 0
        }
        
        try:
            file = await FileRecord.get(file_id)
            if not file:
                return results
            
            # Get Phase 3 extractors
            extractors = ExtractorRegistry.get_for_phase(3, locator=self.locator)
            
            for extractor in extractors:
                try:
                    if extractor.can_process(file):
                        extractor_results = await extractor.process([file])
                        success = extractor_results.get(file._id, False)
                        results["by_extractor"][extractor.name] = success
                        
                except Exception as e:
                    logger.error(f"Phase 3 extractor {extractor.name} failed: {e}")
                    results["errors"] += 1
            
            # Update final state
            file = await FileRecord.get(file_id)  # Refresh
            if file:
                file.processing_state = ProcessingState.COMPLETE
                await file.save()
            
            results["processed"] = True
            
            # Publish event
            await self._publish_progress("phase3.complete", {
                "file_id": file_id_str,
                **results
            })
            
        except Exception as e:
            logger.error(f"Phase 3 error for {file_id}: {e}")
        finally:
            self._phase3_pending.discard(file_id_str)
        
        return results
    
    async def _publish_progress(self, event_type: str, data: dict):
        """Publish progress event via CommandBus."""
        try:
            from src.core.commands.bus import CommandBus
            bus = self.locator.get_system(CommandBus)
            if hasattr(bus, 'publish'):
                await bus.publish(f"processing.{event_type}", data)
        except (KeyError, ImportError, AttributeError):
            pass
    
    # ==================== Query Methods ====================
    
    async def get_pending_count(self) -> dict:
        """Get count of pending items per phase."""
        return {
            "phase2": len(self._phase2_pending),
            "phase3": len(self._phase3_pending)
        }
    
    async def get_files_by_state(
        self, 
        state: ProcessingState,
        limit: int = 100
    ) -> List[FileRecord]:
        """Get files at a specific processing state."""
        return await FileRecord.find(
            {"processing_state": state},
            limit=limit
        ).to_list()
    
    # ==================== File Event Handlers ====================
    
    async def _on_file_created(self, event: dict):
        """
        Handle file.created event - queue for Phase 2 and 3 processing.
        
        Args:
            event: Dict with file_id
        """
        try:
            file_id = event.get("file_id")
            if not file_id:
                return
            
            file_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            
            # Queue Phase 2 processing
            await self.enqueue_phase2([file_id])
            
            logger.debug(f"Queued new file for processing: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue new file: {e}")
    
    async def _on_file_modified(self, event: dict):
        """
        Handle file.modified event - re-process metadata.
        
        Args:
            event: Dict with file_id
        """
        try:
            file_id = event.get("file_id")
            if not file_id:
                return
            
            file_id = ObjectId(file_id) if isinstance(file_id, str) else file_id
            
            # Reset processing state and re-queue
            file = await FileRecord.get(file_id)
            if file:
                file.processing_state = ProcessingState.REGISTERED
                await file.save()
                
                # Queue Phase 2 (which will lead to Phase 3)
                await self.enqueue_phase2([file_id])
                
            logger.debug(f"Queued modified file for re-processing: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue modified file: {e}")
