import asyncio
from typing import Callable, Dict, Any, Coroutine, List
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

        # 3. Start Workers
        self._running = True
        for i in range(3):
             task = asyncio.create_task(self._worker(i))
             self._workers.append(task)
             
        await super().initialize()

    async def shutdown(self):
        self._running = False
        for _ in self._workers:
            self._queue.put_nowait(None)
        await asyncio.gather(*self._workers)
        await super().shutdown()

    def register_handler(self, name: str, coro_func: Callable[..., Coroutine]):
        """Register a handler function by name."""
        self._handlers[name] = coro_func
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
        while self._running:
            try:
                task_id = await self._queue.get()
                if task_id is None: break
                
                record = await TaskRecord.get(task_id)
                if not record or record.status != "pending":
                    self._queue.task_done()
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
                    result = await handler(*record.task_args)
                    record.result = str(result)
                    record.status = "completed"
                    record.progress = 100
                except Exception as e:
                    logger.exception(f"Task {task_id} failed")
                    record.error = str(e)
                    record.status = "failed"
                
                await record.save()
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} crashed: {e}")
                await asyncio.sleep(1) # Backoff
