"""
UCoreFS - Processing Pipeline

Background processing service integrated with TaskSystem.
Manages Phase 2/3 enrichment queues and worker tasks.
"""
from typing import List, Optional, Set, Dict
from bson import ObjectId
from loguru import logger
import asyncio

from src.core.base_system import BaseSystem
from src.core.tasks.system import TaskSystem
from src.ucorefs.models.base import ProcessingState
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.extractors.protocols import IExtractorRegistry


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
        
        # Inject ExtractorRegistry (breaks circular dependency)
        # Import here once during initialization, not at module level
        from src.ucorefs.extractors import ExtractorRegistry
        self._extractor_registry: IExtractorRegistry = ExtractorRegistry
        logger.debug("ExtractorRegistry injected into ProcessingPipeline")
        
        # Register task handlers
        self.task_system.register_handler("process_phase2_batch", self._handle_phase2_batch)
        self.task_system.register_handler("process_phase3_item", self._handle_phase3_item)
        
        # Track pending work to avoid duplicates
        self._phase2_pending: Set[str] = set()
        self._phase3_pending: Set[str] = set()
        
        # AI executor is now managed by TaskSystem (Phase 3 consolidation)
        # Access via self.task_system.get_ai_executor() or self.get_ai_executor()
        
        # Cache extractor configs - build once instead of on every task
        self._extractor_configs = self._build_extractor_configs()
        
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
        
        # AI executor is now managed by TaskSystem, no need to shutdown here
        
        await super().shutdown()
    
    def _build_extractor_configs(self) -> Dict[str, dict]:
        """
        Build extractor configuration dict from config.
        
        Returns:
            Dict keyed by extractor name with config values
        """
        configs = {}
        
        try:
            logger.info("[Pipeline] Building extractor configs...")
            
            # Add database config for process pool (spawned processes need their own DB connection)
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'database'):
                db_config = self.config.data.database
                configs['db_uri'] = getattr(db_config, 'uri', 'mongodb://localhost:27017')
                configs['db_name'] = getattr(db_config, 'name', 'foundation_app')
            else:
                configs['db_uri'] = 'mongodb://localhost:27017'
                configs['db_name'] = 'foundation_app'

            
            # Get detection config
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'processing'):
                processing = self.config.data.processing
                
                if hasattr(processing, 'detection'):
                    detection = processing.detection
                    
                    # YOLO config
                    if hasattr(detection, 'yolo'):
                        yolo_config = detection.yolo.model_dump() if hasattr(detection.yolo, 'model_dump') else {}
                        configs['yolo'] = yolo_config
                        logger.info(f"[Pipeline] YOLO config: {yolo_config}")
                    
                    # GroundingDINO config
                    if hasattr(detection, 'grounding_dino'):
                        gdino_config = detection.grounding_dino.model_dump() if hasattr(detection.grounding_dino, 'model_dump') else {}
                        configs['grounding_dino'] = gdino_config
                        logger.info(f"[Pipeline] GroundingDINO config: {gdino_config}")
                
                # WD Tagger config
                if hasattr(processing, 'wd_tagger'):
                    wd_config = processing.wd_tagger.model_dump() if hasattr(processing.wd_tagger, 'model_dump') else {}
                    configs['wd_tagger'] = wd_config
                    logger.info(f"[Pipeline] WDTagger config: {wd_config}")
                else:
                    logger.warning("[Pipeline] No wd_tagger config found in processing")
            else:
                logger.warning("[Pipeline] No processing config found")
            
            logger.info(f"[Pipeline] Built {len(configs)} extractor configs: {list(configs.keys())}")
        
        except Exception as e:
            logger.error(f"Failed to build extractor configs: {e}", exc_info=True)
        
        return configs
    
    def get_ai_executor(self):
        """
        Get the shared AI thread pool executor from TaskSystem.
        
        Returns:
            ThreadPoolExecutor for CPU-heavy AI preprocessing tasks, or None
        """
        return self.task_system.get_ai_executor() if self.task_system else None
    
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
        
        Split architecture:
        1. CPU-only extractors (thumbnail, metadata) → Process pool
        2. AI extractors (CLIP, DINO, WD-Tagger) → Thread pool (preloaded models)
        """
        file_ids = [ObjectId(fid) for fid in file_ids_str.split(",") if fid]
        
        logger.info(f"[PHASE2_START] Processing batch of {len(file_ids)} files")
        logger.debug(f"[PHASE2_START] File IDs: {[str(fid)[:8] + '...' for fid in file_ids]}")
        
        results = {
            "processed": 0,
            "errors": 0,
            "by_extractor": {}
        }
        
        # STEP 1: CPU-only extractors in process pool
        try:
            from src.ucorefs.processing.process_handlers import process_phase2_batch_sync
            
            logger.info("[PHASE2] Step 1: CPU-only extractors in process pool")
            cpu_results = await self.task_system.run_in_process(
                process_phase2_batch_sync,
                file_ids_str,
                self._extractor_configs
            )
            results["by_extractor"].update(cpu_results.get("by_extractor", {}))
            results["processed"] += cpu_results.get("processed", 0)
            results["errors"] += cpu_results.get("errors", 0)
            
        except RuntimeError as e:
            logger.warning(f"[PHASE2] Process pool not available: {e}")
        
        # STEP 2: AI extractors in thread pool (uses preloaded models)
        logger.info("[PHASE2] Step 2: AI extractors in thread pool")
        await self._run_ai_extractors_phase2(file_ids, results)
        
        # Queue Phase 3 for successfully processed files
        for file_id in file_ids:
            await self.enqueue_phase3(file_id)
        
        # Remove from pending
        for file_id in file_ids:
            self._phase2_pending.discard(str(file_id))
        
        # Publish event
        await self._publish_progress("phase2.complete", results)
        
        return results
    
    async def _run_ai_extractors_phase2(self, file_ids: list, results: dict):
        """Run AI extractors (needs_model=True) in thread pool."""
        # Load files
        files = []
        for file_id in file_ids:
            try:
                file = await FileRecord.get(file_id)
                if file:
                    files.append(file)
            except Exception as e:
                logger.error(f"[PHASE2_AI] Failed to load {file_id}: {e}")
                results["errors"] += 1
        
        if not files:
            return
        
        # Get AI extractors for Phase 2
        all_extractors = self._extractor_registry.get_for_phase(2, locator=self.locator, config=self._extractor_configs)
        ai_extractors = [e for e in all_extractors if getattr(e, 'needs_model', False)]
        
        if not ai_extractors:
            logger.debug("[PHASE2_AI] No AI extractors for Phase 2")
            return
        
        logger.info(f"[PHASE2_AI] Running {len(ai_extractors)} AI extractors: {[e.name for e in ai_extractors]}")
        
        for extractor in ai_extractors:
            try:
                processable = [f for f in files if extractor.can_process(f)]
                
                if processable:
                    extractor_results = await extractor.process(processable)
                    success_count = sum(1 for v in extractor_results.values() if v)
                    results["by_extractor"][extractor.name] = success_count
                    results["processed"] += success_count
                    
                    logger.info(f"[PHASE2_AI] {extractor.name}: {success_count}/{len(processable)} successful")
            except Exception as e:
                logger.error(f"[PHASE2_AI] Extractor {extractor.name} failed: {e}")
                results["errors"] += 1
    
    async def _handle_phase3_item(self, file_id_str: str) -> dict:
        """
        Handle Phase 3 single item processing task.
        
        Operations run via ExtractorRegistry:
        1. BLIP captioning
        2. GroundingDINO object detection
        3. Other Phase 3 extractors
        4. DetectionService (YOLO/MTCNN) if enabled
        """
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
            
            # Get Phase 3 extractors (using injected registry)
            # Extractors now handle LLM vs legacy routing internally via AIExtractor
            extractors = self._extractor_registry.get_for_phase(3, locator=self.locator, config=self._extractor_configs)
            
            for extractor in extractors:
                try:
                    if extractor.can_process(file):
                        # AIExtractor handles LLM worker vs legacy routing internally
                        extractor_results = await extractor.process([file])
                        success = extractor_results.get(file._id, False)
                        results["by_extractor"][extractor.name] = success
                        
                except Exception as e:
                    logger.error(f"Phase 3 extractor {extractor.name} failed: {e}")
                    results["errors"] += 1
            
            # Run DetectionService if configured
            results["detections"] = await self._run_detection(file)
            
            # Update final state ONLY if Phase 3 actually did work
            file = await FileRecord.get(file_id)  # Refresh
            if file:
                # Check if any Phase 3 extractor succeeded OR detections were created
                phase3_success = any(results["by_extractor"].values()) or results["detections"] > 0
                
                if phase3_success:
                    old_state = file.processing_state
                    file.processing_state = ProcessingState.COMPLETE
                    await file.save()
                    # Handle both enum and int for old_state
                    old_state_name = old_state.name if hasattr(old_state, 'name') else ProcessingState(old_state).name
                    logger.info(f"[PHASE3_TRANSITION] ✓ {file.name}: {old_state_name} → COMPLETE")
                    logger.debug(f"[PHASE3_TRANSITION]   Phase 3 extractors: {list(results['by_extractor'].keys())}")
                    logger.debug(f"[PHASE3_TRANSITION]   Detections: {results['detections']}")
                else:
                    # Phase 3 had nothing to do or all failed - stay at current state
                    # Handle both enum and int for processing_state
                    state_name = file.processing_state.name if hasattr(file.processing_state, 'name') else ProcessingState(file.processing_state).name
                    logger.warning(f"[PHASE3_TRANSITION] ⚠ {file.name}: Phase 3 had no work or all failed (state={state_name} unchanged)")
                    logger.debug(f"[PHASE3_TRANSITION]   Extractor results: {results['by_extractor']}")
                
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
            
            # SAN-14: Check if backend is actually available/loaded
            if not detection_service.get_backend(backend):
                # Only log once per session ideally, or debug to avoid spam
                logger.debug(f"Detection backend '{backend}' configured but not available - skipping")
                return 0

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
