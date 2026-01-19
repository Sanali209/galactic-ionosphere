import asyncio
import uuid
from typing import Callable, Dict, Any, Coroutine, List, Union, Awaitable, Optional
from functools import partial
from loguru import logger

from PySide6.QtCore import QObject, Signal

from ..base_system import BaseSystem
from .models import TaskRecord
from .runner import BackgroundTaskRunner

class TaskSystemSignals(QObject):
    """Qt signals for TaskSystem (composition pattern to avoid metaclass conflict)."""
    task_started = Signal(str)              # task_id
    task_completed = Signal(str, object)    # task_id, result
    task_failed = Signal(str, str)          # task_id, error_message
    task_progress = Signal(str, int, str)   # task_id, percent, message


class TaskSystem(BaseSystem):
    """
    Background task execution system.
    
    Manages task queue and worker pool for async task execution.
    Integrated with Foundation's ServiceLocator and event bus.
    
    New: Supports non-blocking background execution via BackgroundTaskRunner.
    Use submit_background() for tasks that should not block UI.
    """
    
    # Shutdown sentinel for PriorityQueue (must be comparable with tuples)
    SHUTDOWN_SENTINEL = (-1, None)
    # SAN-14 Phase 3: Priority constants
    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = 1
    PRIORITY_LOW = 2
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._queue = asyncio.PriorityQueue()  # SAN-14 Phase 3: Priority queue
        self._workers = []
        self._running = False
        self._handlers: Dict[str, Callable] = {}
        
        # Qt signals (via composition to avoid metaclass conflict)
        self.signals = TaskSystemSignals()
        
        # Background task runner (QThread with own asyncio loop)
        self._bg_runner: Optional[BackgroundTaskRunner] = None
        
        # Process pool for CPU-heavy non-LLM tasks (Phase 2)
        self._process_executor: Optional["ProcessExecutor"] = None
        
        # Thread pool for AI inference (consolidated from pipeline)
        self._ai_executor: Optional["ThreadPoolExecutor"] = None
        
        # Runtime cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        logger.info("TaskSystem initializing...")
        
        from datetime import datetime
        now = int(datetime.utcnow().timestamp())
        
        # 1. Recovery: Reset 'running' tasks to 'pending' with crash metadata
        try:
            coll = TaskRecord.get_collection()
            
            running_count = await coll.count_documents({"status": "running"})
            if running_count > 0:
                logger.info(f"📊 RECOVERY: Found {running_count} tasks with status 'running'")
            
            if running_count > 0:
                # Reset with interruption metadata and increment recovery counter
                result = await coll.update_many(
                    {"status": "running"},
                    {
                        "$set": {
                            "status": "pending", 
                            "interrupted": "System crash/restart",
                            "updated_at": now
                        },
                        "$inc": {"recovery_count": 1}
                    }
                )
                logger.info(f"✅ RECOVERY: Reset {result.modified_count} tasks from 'running' to 'pending'")
                
        except Exception as e:
            logger.error(f"❌ RECOVERY FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # 1b. Detect and quarantine poison tasks (failed 3+ times)
        try:
            coll = TaskRecord.get_collection()
            
            poison_count = await coll.count_documents({"recovery_count": {"$gte": 3}})
            if poison_count > 0:
                logger.critical(f"⚠️ POISON: Found {poison_count} tasks that crashed 3+ times - quarantining")
                await coll.update_many(
                    {"recovery_count": {"$gte": 3}},
                    {"$set": {"status": "quarantined", "error": "Too many failures (poison task)", "updated_at": now}}
                )
        except Exception as e:
            logger.error(f"❌ Poison task detection failed: {e}")
        
        # 1c. Aggressive cleanup for failed tasks - delete old ones (>1 hour) to prevent bloat
        # No auto-retry: failed tasks stay failed until manually reset
        try:
            coll = TaskRecord.get_collection()
            failed_retention_seconds = 3600  # 1 hour
            failed_cutoff = now - failed_retention_seconds
            
            # Delete old failed tasks (non-retryable behavior - extractor failures shouldn't auto-retry)
            old_failed = await coll.delete_many({
                "status": "failed",
                "updated_at": {"$lt": failed_cutoff}
            })
            
            if old_failed.deleted_count > 0:
                logger.info(f"🧹 CLEANUP: Deleted {old_failed.deleted_count} old failed tasks (>1h)")
            
            # Count remaining failed tasks for visibility
            remaining_failed = await coll.count_documents({"status": "failed"})
            if remaining_failed > 0:
                logger.warning(f"⚠️ {remaining_failed} recent failed tasks remain (will NOT auto-retry)")
                
        except Exception as e:
            logger.error(f"❌ Failed task cleanup error: {e}")
        
        # 1d. TTL-based cleanup for completed tasks (keep 24h history)
        try:
            coll = TaskRecord.get_collection()
            completed_retention_seconds = 24 * 3600  # 24 hours
            completed_cutoff = now - completed_retention_seconds
            
            deleted = await coll.delete_many({
                "status": "completed",
                "completed_at": {"$lt": completed_cutoff}
            })
            if deleted.deleted_count > 0:
                logger.info(f"🧹 CLEANUP: Removed {deleted.deleted_count} completed tasks older than 24h")
        except Exception as e:
            logger.debug(f"Completed task cleanup skipped: {e}")

        
        # 2. Reload ONLY 'pending' tasks (not failed, not running)
        # Use simple count first to avoid loading massive objects if not needed immediately
        # The workers pull from DB as needed, we just need to queue IDs
        
        # LOGGING FOR VISIBILITY
        db_count = await TaskRecord.get_collection().count_documents({"status": "pending"})
        logger.info(f"📊 DATABASE: {db_count} pending tasks exist in DB")
        
        pending_cursor = TaskRecord.get_collection().find({"status": "pending"})
        # Sort by priority (0=High) then creation time
        pending_cursor.sort([("priority", 1), ("created_at", 1)])
        
        count = 0
        async for task_data in pending_cursor:
            # We only need ID and priority for the queue
            tid = task_data["_id"]
            # Read priority from DB (default to NORMAL if missing for backward compat)
            task_priority = task_data.get("priority", self.PRIORITY_NORMAL)
            await self._queue.put((task_priority, tid))
            count += 1
            
        logger.info(f"📥 RESTORED: {count} pending tasks to execution queue (queue size: {self._queue.qsize()})")
        
        # Initialize process pool for CPU-heavy non-LLM tasks
        process_workers = 4
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'processing'):
            process_workers = getattr(self.config.data.processing, 'process_workers', 4)
        
        if process_workers > 0:
            from .process_executor import ProcessExecutor
            self._process_executor = ProcessExecutor(max_workers=process_workers)
            logger.info(f"Process pool initialized with {process_workers} workers")
        
        # Initialize AI thread pool for inference (consolidated from pipeline)
        ai_workers = 8
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'processing'):
            ai_workers = getattr(self.config.data.processing, 'ai_workers', 8)
        
        if ai_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._ai_executor = ThreadPoolExecutor(
                max_workers=ai_workers,
                thread_name_prefix="ai-cpu"
            )
            logger.info(f"AI thread pool initialized with {ai_workers} workers")
        
        # Subscribe to config changes for reactive updates
        if hasattr(self.config, 'on_changed'):
            self.config.on_changed.connect(self._on_config_changed)
            
        await super().initialize()
        
    async def start_workers(self):
        """Start task workers and background runner."""
        if self._running:
            return
            
        # 3. Start Workers (read count from config)
        worker_count = 8
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'general'):
            worker_count = getattr(self.config.data.general, 'task_workers', 8)
        
        self._running = True
        for i in range(worker_count):
             task = asyncio.create_task(self._worker(i))
             self._workers.append(task)
             
        logger.info(f"TaskSystem started {worker_count} background workers on TaskSystem {id(self)}")
        
        # Start BackgroundTaskRunner (QThread for non-blocking execution)
        if not self._bg_runner:
            self._bg_runner = BackgroundTaskRunner(task_system=self)
            # Forward runner signals to TaskSystem signals
            self._bg_runner.task_started.connect(self.signals.task_started.emit)
            self._bg_runner.task_completed.connect(self.signals.task_completed.emit)
            self._bg_runner.task_failed.connect(self.signals.task_failed.emit)
            self._bg_runner.task_progress.connect(self.signals.task_progress.emit)
            # Handlers are now accessed directly from TaskSystem (no mirroring needed)
            self._bg_runner.start()
            logger.info("BackgroundTaskRunner started")
        
        # Start runtime cleanup scheduler (runs every hour)
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Runtime cleanup scheduler started")
        
        # Log registered handlers for visibility
        handlers = self.get_registered_handlers()
        if handlers:
            logger.info(f"📋 Registered {len(handlers)} task handlers: {handlers}")
            
        return worker_count

    async def shutdown(self):
        if not self._running:
            return # Already shut down
            
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        # Stop BackgroundTaskRunner
        if self._bg_runner and self._bg_runner.isRunning():
            self._bg_runner.stop()
            logger.info("BackgroundTaskRunner stopped")
        
        # Shutdown process pool
        if self._process_executor:
            self._process_executor.shutdown(wait=True)
        
        # Shutdown AI thread pool
        if self._ai_executor:
            self._ai_executor.shutdown(wait=False)
            logger.info("AI thread pool shutdown")
        
        # Send shutdown sentinel to each worker (must be comparable with tuples in PriorityQueue)
        for _ in self._workers:
            self._queue.put_nowait(self.SHUTDOWN_SENTINEL)
        await asyncio.gather(*self._workers)
        await super().shutdown()

    def _on_config_changed(self, section: str, key: str, value):
        """Handle config changes for reactive updates."""
        if section == "general" and key == "task_workers":
            logger.info(f"Task worker count changed to {value} (restart required)")
            # Note: Dynamic worker adjustment would require complex queue management
            # For now, log the change - restart needed for new count

    async def _periodic_cleanup(self):
        """Run cleanup every hour during runtime to prevent task record bloat."""
        cleanup_interval = 3600  # 1 hour
        
        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)
                
                if not self._running:
                    break
                
                from datetime import datetime
                now = int(datetime.utcnow().timestamp())
                coll = TaskRecord.get_collection()
                
                # Cleanup completed tasks older than 24h
                completed_cutoff = now - (24 * 3600)
                completed_result = await coll.delete_many({
                    "status": "completed",
                    "completed_at": {"$lt": completed_cutoff}
                })
                
                # Cleanup failed non-retryable tasks older than 1h
                failed_cutoff = now - 3600
                failed_result = await coll.delete_many({
                    "status": "failed",
                    "retryable": {"$ne": True},
                    "completed_at": {"$lt": failed_cutoff}
                })
                
                total_cleaned = completed_result.deleted_count + failed_result.deleted_count
                if total_cleaned > 0:
                    logger.info(f"🧹 Runtime cleanup: Deleted {total_cleaned} old tasks (completed: {completed_result.deleted_count}, failed: {failed_result.deleted_count})")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Runtime cleanup error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def register_handler(self, name: str, func: Union[Callable[..., Coroutine], Callable[..., Any]]):
        """Register a handler function by name."""
        self._handlers[name] = func
        # BackgroundTaskRunner now accesses handlers directly via _get_handler()
        logger.debug(f"Registered task handler: {name}")
    
    def get_registered_handlers(self) -> List[str]:
        """
        Get list of all registered handler names.
        
        Returns:
            List of handler names for debugging/introspection.
        """
        return list(self._handlers.keys())
    
    def has_handler(self, name: str) -> bool:
        """
        Check if a handler is registered.
        
        Args:
            name: Handler name to check
            
        Returns:
            True if handler exists
        """
        return name in self._handlers
    
    def submit_background(self, handler_name: str, *args, 
                          priority: int = None, **kwargs) -> str:
        """
        Submit a task for non-blocking background execution.
        
        Task runs in separate QThread - UI never blocks.
        Result delivered via signals.task_completed signal.
        
        Usage:
            task_id = task_system.submit_background("handler", arg1, arg2)
            task_system.signals.task_completed.connect(self.on_result)
            
            def on_result(self, task_id: str, result: Any):
                self.update_ui(result)
        
        Args:
            handler_name: Name of registered handler
            *args: Arguments for handler
            priority: 0=HIGH, 1=NORMAL, 2=LOW
            **kwargs: Keyword arguments for handler
            
        Returns:
            task_id for tracking
        """
        if handler_name not in self._handlers:
            raise ValueError(f"Unknown handler: {handler_name}")
        
        if not self._bg_runner or not self._bg_runner.is_running():
            logger.warning("BackgroundTaskRunner not running, task may be delayed")
        
        if priority is None:
            priority = self.PRIORITY_NORMAL
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Submit to BackgroundTaskRunner
        self._bg_runner.submit(handler_name, task_id, *args, priority=priority, **kwargs)
        
        logger.debug(f"Background task submitted: {task_id[:8]}... ({handler_name})")
        return task_id
    
    async def run_in_process(self, func: Callable, *args, **kwargs):
        """
        Run function in separate process (non-blocking).
        
        Bypasses GIL for CPU-intensive work. Use for Phase 2 batch processing
        (thumbnails, EXIF extraction, CLIP embeddings).
        
        Args:
            func: Top-level picklable function
            *args: Picklable arguments
            **kwargs: Picklable keyword arguments
            
        Returns:
            Result from func execution
            
        Raises:
            RuntimeError: If process pool not initialized
        """
        if not self._process_executor:
            raise RuntimeError("Process pool not initialized")
        
        return await self._process_executor.submit(func, *args, **kwargs)
    
    def get_ai_executor(self):
        """
        Get the shared AI thread pool executor.
        
        Returns:
            ThreadPoolExecutor for AI inference tasks, or None if not initialized.
            
        Usage by extractors:
            executor = task_system.get_ai_executor()
            result = await loop.run_in_executor(executor, self._inference_sync, ...)
        """
        return self._ai_executor
    
    async def submit(self, handler_name: str, task_name: str, *args, priority: int = None):
        """
        Submit a new task.
        
        Args:
            handler_name: Registered name of the function to call
            task_name: Human readable title
            *args: Arguments to pass to the handler (must be serializable)
            priority: Task priority (0=HIGH, 1=NORMAL, 2=LOW). Default: NORMAL
        
        Returns:
            Task ID
        """
        # --- Smart Routing (Client -> Engine) ---
        # If we are in the UI (Client) thread and have access to EngineProxy,
        # we route this execution to the Engine Thread where the workers are.
        try:
            from src.core.engine.proxy import EngineProxy
            if self.locator.has_system(EngineProxy):
                proxy = self.locator.get_system(EngineProxy)
                if proxy and proxy._is_ready: # Only route if engine is running
                    
                    # Define closure to run inside Engine Thread
                    async def _engine_submit():
                        try:
                            from PySide6.QtCore import QThread
                            thread = QThread.currentThread()
                            # Reliable way to get Engine Locator
                            if hasattr(thread, 'locator') and thread.locator:
                                engine_sl = thread.locator
                            else:
                                from src.core.locator import get_active_locator
                                logger.warning("TaskSystem Routing: QThread.locator missing, falling back to ContextVar")
                                engine_sl = get_active_locator() 
                            
                            engine_ts = engine_sl.get_system(TaskSystem)
                            # Call recursively (will hit the "Engine Mode" path below)
                            return await engine_ts.submit(handler_name, task_name, *args, priority=priority)
                        except Exception as e:
                            logger.error(f"TaskSystem Routing Closure Error: {e}", exc_info=True)
                            raise
                    
                    logger.info(f"TaskSystem: Routing task '{task_name}' to Engine")
                    
                    # Dispatch to Engine Thread and wait for result (Task ID)
                    # proxy.submit returns a concurrent.futures.Future
                    future = proxy.submit(_engine_submit())
                    task_id = await asyncio.wrap_future(future)
                    return task_id
                else:
                    logger.warning(f"TaskSystem Routing: EngineProxy present but not ready (proxy={proxy}, ready={getattr(proxy, '_is_ready', 'N/A')})")
            else:
                 # Normal for Engine Thread (no Proxy)
                 pass
        except (ImportError, KeyError) as e:
            # Normal for Engine Thread (ImportError if proxy not avail, KeyError if system not found)
            pass 
        except Exception as e:
            logger.error(f"TaskSystem Routing Failed: {e}")
        # ----------------------------------------

        if handler_name not in self._handlers:
            raise ValueError(f"Unknown handler: {handler_name}")
        
        # SAN-14 Phase 3: Default priority
        if priority is None:
            priority = self.PRIORITY_NORMAL
        
        # Serialize args
        str_args = [str(a) for a in args]

        record = TaskRecord(
            name=task_name, 
            handler_name=handler_name,
            task_args=str_args,
            status="pending",
            priority=priority  # Persist priority to DB
        )
        await record.save()
        
        # SAN-14 Phase 3: Queue with priority
        await self._queue.put((priority, record.id))
        
        logger.info(f"Task submitted: {task_name} ({record.id}) [priority={priority}] on TaskSystem {id(self)}")
        return record.id

    async def _queue_task(self, record: TaskRecord, priority: int = None):
        """Queue a task with optional priority (used for recovery)."""
        if priority is None:
            priority = self.PRIORITY_NORMAL
        await self._queue.put((priority, record.id))

    async def _worker(self, worker_id: int):
        """Worker task that processes items from the queue."""
        logger.debug(f"TaskSystem worker {worker_id} started")
        consecutive_failures = 0
        max_consecutive_failures = 5
        base_backoff_seconds = 1.0
        
        while self._running:
            try:
                # Get next item (blocks until available)
                queue_item = await self._queue.get()
                
                # Check for shutdown sentinel (comparable tuple)
                if queue_item == self.SHUTDOWN_SENTINEL or queue_item[1] is None:
                    logger.debug(f"Worker {worker_id} received shutdown sentinel")
                    break
                
                # Unpack priority tuple
                priority, task_id = queue_item
                logger.debug(f"Worker {worker_id} popped task {task_id} (p={priority})")
                
                record = await TaskRecord.get(task_id)
                if not record:
                    logger.warning(f"Worker {worker_id}: Task {task_id} not found in DB")
                    self._queue.task_done()
                    continue
                    
                if record.status != "pending":
                    logger.debug(f"Worker {worker_id}: Task {task_id} status is '{record.status}', skipping")
                    self._queue.task_done()
                    consecutive_failures = 0
                    continue
                
                handler = self._handlers.get(record.handler_name)
                if not handler:
                    record.status = "failed"
                    record.error = f"Handler '{record.handler_name}' not found"
                    await record.save()
                    self._queue.task_done()
                    continue

                # Execute
                from datetime import datetime
                now = int(datetime.utcnow().timestamp())
                record.status = "running"
                record.started_at = now
                record.updated_at = now
                await record.save()
                
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(*record.task_args)
                    else:
                        loop = asyncio.get_running_loop()
                        from functools import partial
                        result = await loop.run_in_executor(None, partial(handler, *record.task_args))

                    record.status = "completed"
                    record.result = result
                    record.completed_at = int(datetime.utcnow().timestamp())
                    record.updated_at = int(datetime.utcnow().timestamp())
                    consecutive_failures = 0
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    
                    # Categorize error for retry logic
                    if "FileNotFoundError" in error_msg or "not found" in error_msg.lower():
                        error_type = "missing_resource"
                        retryable = False
                    elif "ConnectionError" in error_msg or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                        error_type = "network"
                        retryable = True
                    else:
                        error_type = "unknown"
                        retryable = False
                    
                    record.status = "failed"
                    record.error = error_msg
                    record.error_type = error_type
                    record.retryable = retryable
                    record.completed_at = int(datetime.utcnow().timestamp())
                    record.updated_at = int(datetime.utcnow().timestamp())
                    logger.error(f"Task {record.id} failed ({error_type}): {e}\n{traceback.format_exc()}")
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        backoff_time = base_backoff_seconds * (2 ** (consecutive_failures - max_consecutive_failures))
                        logger.warning(f"Worker {worker_id} backing off for {backoff_time}s due to consecutive failures")
                        await asyncio.sleep(backoff_time)

                await record.save()
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"Worker {worker_id} hit max consecutive failures "
                        f"({max_consecutive_failures}), entering extended backoff"
                    )
                
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
                backoff = min(base_backoff_seconds * (2 ** (consecutive_failures - 1)), 30.0)
                logger.debug(f"Worker {worker_id} backing off for {backoff:.1f}s")
                
                try:
                    await asyncio.sleep(backoff)
                except (RuntimeError, asyncio.CancelledError):
                    # Loop closed or cancelled during sleep
                    break

