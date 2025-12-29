import asyncio
from typing import Callable, Dict, Any, Coroutine, List, Union, Awaitable
from functools import partial
from loguru import logger
from ..base_system import BaseSystem
from .models import TaskRecord

class TaskSystem(BaseSystem):
    """
    Background task execution system.
    
    Manages task queue and worker pool for async task execution.
    Integrated with Foundation's ServiceLocator and event bus.
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

    async def initialize(self):
        logger.info("TaskSystem initializing...")
        
        # 1. Recovery: Reset 'running' tasks to 'pending'
        # In a real app we might want to check if they are actually dead or on another node
        running_tasks = await TaskRecord.find({"status": "running"})
        if running_tasks:
            logger.warning(f"Found {len(running_tasks)} interrupted tasks. Rescheduling...")
            for task in running_tasks:
                task.status = "pending"
                task.error = "Interrupted by system restart"
                await task.save()

        # 2. Reload 'pending' tasks
        pending_tasks = await TaskRecord.find({"status": "pending"})
        for task in pending_tasks:
            await self._queue_task(task)
        logger.info(f"Loaded {len(pending_tasks)} pending tasks.")

        # 3. Start Workers (read count from config)
        worker_count = 8  # default (increased from 3 for better concurrency - SAN-14)
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'general'):
            worker_count = getattr(self.config.data.general, 'task_workers', 8)
        
        self._running = True
        for i in range(worker_count):
             task = asyncio.create_task(self._worker(i))
             self._workers.append(task)
        logger.info(f"Started {worker_count} task workers")
        
        # Subscribe to config changes for reactive updates
        if hasattr(self.config, 'on_changed'):
            self.config.on_changed.connect(self._on_config_changed)
             
        await super().initialize()

    async def shutdown(self):
        self._running = False
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

    def register_handler(self, name: str, func: Union[Callable[..., Coroutine], Callable[..., Any]]):
        """Register a handler function by name."""
        self._handlers[name] = func
        logger.debug(f"Registered task handler: {name}")

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
            status="pending"
        )
        await record.save()
        
        # SAN-14 Phase 3: Queue with priority
        await self._queue.put((priority, record.id))
        
        logger.info(f"Task submitted: {task_name} ({record.id}) [priority={priority}]")
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
                    break
                
                # Unpack priority tuple
                priority, task_id = queue_item
                
                record = await TaskRecord.get(task_id)
                if not record or record.status != "pending":
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
                record.status = "running"
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
                    consecutive_failures = 0
                except Exception as e:
                    import traceback
                    record.status = "failed"
                    record.error = str(e)
                    logger.error(f"Task {record.id} failed: {e}\n{traceback.format_exc()}")
                    
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

