"""
UCoreFS - Processing Pipeline

Background processing service integrated with TaskSystem.
Manages Phase 2/3 enrichment queues and worker tasks.
"""
from typing import List, Optional, Set
from bson import ObjectId
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    
    # Dependencies for task submission
    depends_on = ["TaskSystem", "DatabaseManager"]
    
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
        
        # SAN-14 Phase 2: Create dedicated AI thread pool for CPU-heavy preprocessing
        # Default to same count as general task workers (often 8) to maximize throughput
        default_workers = 8
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'general'):
             default_workers = getattr(self.config.data.general, 'task_workers', 8)

        ai_workers = default_workers
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'processing'):
            # Override if specifically set in processing config, otherwise use general count
            configured = getattr(self.config.data.processing, 'ai_workers', None)
            if configured:
                ai_workers = configured
        
        self._ai_executor = ThreadPoolExecutor(
            max_workers=ai_workers,
            thread_name_prefix="ai-cpu"
        )
        logger.info(f"Created dedicated AI thread pool with {ai_workers} workers")
        
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
        
        # SAN-14 Phase 2: Shutdown AI executor
        if hasattr(self, '_ai_executor') and self._ai_executor:
            logger.info("Shutting down AI thread pool...")
            self._ai_executor.shutdown(wait=True)
        
        await super().shutdown()
    
    def get_ai_executor(self):
        """
        Get the dedicated AI thread pool executor.
        
        Returns:
            ThreadPoolExecutor for CPU-heavy AI preprocessing tasks, or None
        """
        return getattr(self, '_ai_executor', None)
    
    # ==================== Enqueue Methods ====================
    
    async def enqueue_phase2(self, file_ids: List[ObjectId], force: bool = False, priority: int = None) -> Optional[str]:
        """
        Enqueue files for Phase 2 processing.
        
        Args:
            file_ids: List of file ObjectIds to process
            force: If True, ignore pending check and always process
            priority: Task priority (0=HIGH, 1=NORMAL, 2=LOW). Default: NORMAL
            
        Returns:
            Task ID if submitted, None if already pending (and not forced)
        """
        if force:
            # Force mode: add all files
            new_ids = list(file_ids)
        else:
            # Filter out already pending
            new_ids = [fid for fid in file_ids if str(fid) not in self._phase2_pending]
        
        if not new_ids:
            return None
        
        # Mark as pending
        for fid in new_ids:
            self._phase2_pending.add(str(fid))
        
        # Submit tasks in batches
        last_task_id = None
        count_queued = 0
        
        for i in range(0, len(new_ids), self.PHASE2_BATCH_SIZE):
            batch = new_ids[i:i + self.PHASE2_BATCH_SIZE]
            batch_str = ",".join(str(fid) for fid in batch)
            
            # SAN-14 Phase 3: Pass priority through
            task_id = await self.task_system.submit(
                "process_phase2_batch",
                f"Phase 2: Process {len(batch)} files",
                batch_str,
                priority=priority  # Pass priority
            )
            last_task_id = task_id
            count_queued += 1
            
            logger.debug(f"Enqueued Phase 2 batch {count_queued} (task {task_id}) for {len(batch)} files")

        logger.info(f"Enqueued {count_queued} Phase 2 tasks for {len(new_ids)} total files")
        return last_task_id
    
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
            file_id_str,
            priority=0 # TaskSystem.PRIORITY_HIGH
        )
        
        logger.info(f"Enqueued Phase 3 task {task_id} for {file_id_str}")
        return task_id
    
    async def reindex_all(self, include_processed: bool = False) -> dict:
        """
        Reindex all files in database via Phase 2 processing.
        
        Args:
            include_processed: If True, reprocess all files. 
                              If False, only files with processing_state < READY.
        
        Returns:
            Dict with stats: {"total_files", "batches_queued", "task_ids"}
        """
        from src.ucorefs.models.base import ProcessingState
        
        # Build query
        if include_processed:
            query = {}  # All files
        else:
            # Files not fully processed
            query = {"processing_state": {"$lt": ProcessingState.AI_READY.value}}
        
        # Get all file IDs
        files = await FileRecord.find(query)
        file_ids = [f._id for f in files]
        
        if not file_ids:
            logger.info("Reindex: No files to process")
            return {"total_files": 0, "batches_queued": 0, "task_ids": []}
        
        logger.info(f"Reindex: Found {len(file_ids)} files to process")
        
        # Batch into Phase 2 tasks
        task_ids = []
        for i in range(0, len(file_ids), self.PHASE2_BATCH_SIZE):
            batch = file_ids[i:i + self.PHASE2_BATCH_SIZE]
            task_id = await self.enqueue_phase2(batch, force=True)
            if task_id:
                task_ids.append(task_id)
        
        result = {
            "total_files": len(file_ids),
            "batches_queued": len(task_ids),
            "task_ids": task_ids
        }
        
        logger.info(f"Reindex: Queued {len(task_ids)} batches for {len(file_ids)} files")
        return result
    
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
        total_extractors = len(extractors)
        
        # SAN-14 Phase 2: Track progress per extractor
        for i, extractor in enumerate(extractors):
            try:
                # Filter to files this extractor can process
                processable = [f for f in files if extractor.can_process(f)]
                
                if processable:
                    logger.info(f"Running {extractor.name} on {len(processable)} files")
                    
                    extractor_results = await extractor.process(processable)
                    success_count = sum(1 for v in extractor_results.values() if v)
                    results["by_extractor"][extractor.name] = success_count
                    results["processed"] += success_count
                    
                    # SAN-14 Phase 2: Publish per-extractor progress
                    progress_percent = int((i + 1) / total_extractors * 100)
                    await self._publish_progress("phase2.extractor.complete", {
                        "extractor": extractor.name,
                        "processed": len(processable),
                        "success": success_count,
                        "progress": progress_percent,
                        "batch_size": len(files)
                    })
                    
            except Exception as e:
                logger.error(f"Extractor {extractor.name} failed: {e}")
                results["errors"] += 1
        
        # Update state and queue for Phase 3
        from src.ucorefs.models.base import ProcessingState
        
        for file in files:
            # If at least one extractor succeeded, or if no extractors failed critically
            # For now, we assume if it passed Phase 2 extractors, it's ready for Phase 3
            # We can be more strict if needed (e.g. check "processed" count)
            
            try:
                # Update state
                # Note: We re-fetch or assume file object is fresh enough. 
                # Ideally we should use atomic updates if multiple workers touch same file, 
                # but currently pipelines are linear per file.
                
                # We set to INDEXED (40), which implies Phase 2 complete
                file.processing_state = ProcessingState.INDEXED
                await file.save()
                
                # Queue Phase 3
                await self.enqueue_phase3(file._id)
                
            except Exception as e:
                logger.error(f"Failed to transition {file._id} to Phase 3: {e}")
        
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
        4. DetectionService (YOLO/MTCNN) if enabled
        """
        from src.ucorefs.extractors import ExtractorRegistry
        
        file_id = ObjectId(file_id_str)
        
        results = {
            "processed": False,
            "by_extractor": {},
            "detections": 0,
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
            
            # Run DetectionService if configured
            results["detections"] = await self._run_detection(file)
            
            # Update final state
            file = await FileRecord.get(file_id)  # Refresh
            if file:
                file.processing_state = ProcessingState.COMPLETE
                await file.save()
                
                # SAN-14 Phase 3: Update Vector Index immediately
                if "clip" in file.embeddings:
                    try:
                        from src.ucorefs.vectors.faiss_service import FAISSIndexService
                        faiss_service = self.locator.get_system(FAISSIndexService)
                        
                        vector_data = file.embeddings["clip"]
                        # We need the vector itself. FileRecord.embeddings usually stores metadata, 
                        # actual vector is in EmbeddingRecord.
                        
                        from src.ucorefs.vectors.models import EmbeddingRecord
                        emb_record = await EmbeddingRecord.find_one({"file_id": file._id, "provider": "clip"})
                        if emb_record:
                            await faiss_service.add_vector("clip", file._id, emb_record.vector)
                            
                    except Exception as ve:
                        logger.warning(f"Failed to update vector index for {file._id}: {ve}")
            
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
    
    async def _run_detection(self, file: FileRecord) -> int:
        """
        Run detection on a file if DetectionService is configured.
        
        Args:
            file: FileRecord to process
            
        Returns:
            Number of detections found
        """
        try:
            from src.ucorefs.detection import DetectionService
            
            # Check if detection is enabled in config
            config = self.locator.config
            # Access Pydantic models directly via .data
            try:
                detection_config = config.data.processing.detection
            except AttributeError:
                # Fallback if config structure invalid
                return 0
            
            if not detection_config.enabled:
                return 0
            
            # Get DetectionService
            try:
                detection_service = self.locator.get_system(DetectionService)
            except KeyError:
                logger.debug("DetectionService not registered")
                return 0
            
            # Check file type compatibility (images only for now)
            if not file.extension or file.extension.lower() not in [
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
            ]:
                return 0
            
            # Run detection with configured backend
            backend = detection_config.backend
            detections = await detection_service.detect(file._id, backend=backend)
            
            return len(detections)
            
        except ImportError:
            logger.debug("Detection module not available")
            return 0
        except Exception as e:
            logger.error(f"Detection failed for {file._id}: {e}")
            return 0
    
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
