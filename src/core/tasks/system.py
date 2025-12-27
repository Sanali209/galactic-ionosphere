import asyncio
from typing import Callable, Dict, Any, Coroutine, List, Union, Awaitable
from functools import partial
from loguru import logger
from ..base_system import BaseSystem
from .models import TaskRecord

class TaskSystem(BaseSystem):
    """
    Manages background tasks with persistence and crash recovery.
    """
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._queue = asyncio.Queue()
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
        worker_count = 3  # default
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'general'):
            worker_count = getattr(self.config.data.general, 'task_workers', 3)
        
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
        for _ in self._workers:
            self._queue.put_nowait(None)
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

    async def submit(self, handler_name: str, task_name: str, *args):
        """
        Submit a new task.
        :param handler_name: Registered name of the function to call.
        :param task_name: Human readable title.
        :param args: Arguments to pass to the handler (must be serializable).
        """
        if handler_name not in self._handlers:
            raise ValueError(f"Unknown handler: {handler_name}")
            
        # Serialize args (assuming simple types for this template)
        # For complex types, one might use pickle or json dumps
        # Here we just store them as is if ORM ListField supports it, 
        # but our defined ListField(StringField()) only supports strings.
        # Let's assume generic ListField for mixed types or convert to str.
        # Implemented TaskRecord defines ListField(StringField()), so args must be strings.
        str_args = [str(a) for a in args]

        record = TaskRecord(
            name=task_name, 
            handler_name=handler_name,
            task_args=str_args,
            status="pending"
        )
        await record.save()
        await self._queue_task(record)
        logger.info(f"Task submitted: {task_name} ({record.id})")
        return record.id

    async def _queue_task(self, record: TaskRecord):
        await self._queue.put(record.id)

    async def _worker(self, worker_id: int):
        logger.debug(f"Worker {worker_id} started.")
        consecutive_failures = 0
        max_consecutive_failures = 5
        base_backoff_seconds = 1.0
        
        while self._running:
            try:
                task_id = await self._queue.get()
                if task_id is None: break
                
                record = await TaskRecord.get(task_id)
                if not record or record.status != "pending":
                    self._queue.task_done()
                    consecutive_failures = 0  # Reset on successful queue operation
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
                        # Offload synchronous handler to thread
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,  # Use default executor
                            partial(handler, *record.task_args)
                        )
                    
                    record.result = str(result)
                    record.status = "completed"
                    record.progress = 100
                except Exception as e:
                    logger.exception(f"Task {task_id} failed")
                    record.error = str(e)
                    record.status = "failed"
                
                await record.save()
                self._queue.task_done()
                consecutive_failures = 0  # Reset on successful task processing
                
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

