import asyncio
import logging
from typing import Dict, Callable, Any
from datetime import datetime
from src.core.database.models.task import TaskRecord

logger = logging.getLogger(__name__)

class TaskDispatcher:
    """
    Manages background task execution.
    Should be initialized as a Singleton in ServiceLocator.
    """
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._worker_task = None
        
    def register_handler(self, task_type: str, handler: Callable):
        """
        Register a coroutine function to handle a specific task type.
        Handler signature: async def handler(task: TaskRecord) -> Dict (result)
        """
        self._handlers[task_type] = handler

    async def submit_task(self, task_type: str, payload: Dict[str, Any]) -> TaskRecord:
        """Create and save a new task."""
        task = TaskRecord(task_type=task_type)
        task.payload = payload
        task.created_at = datetime.now().timestamp()
        await task.save()
        return task

    async def start(self):
        """Start the worker loop."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Task Dispatcher started.")

    async def stop(self):
        """Stop the worker loop."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Task Dispatcher stopped.")

    async def _worker_loop(self):
        """Polls DB for pending tasks."""
        while self._running:
            try:
                # Find one pending task (FIFO by created_at ideally, but default sort is implicit)
                # We need a proper find_one_and_update or just find then lock logic.
                # For simplicity in this local app: Find one 'pending'.
                
                # We use find() which returns list, take first. 
                # Ideally sort by created_at.
                pending_tasks = await TaskRecord.find({"status": "pending"})
                
                if not pending_tasks:
                    await asyncio.sleep(1) # Wait if queue empty
                    continue
                
                task = pending_tasks[0]
                await self._process_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5) # Backoff on system error

    async def _process_task(self, task: TaskRecord):
        handler = self._handlers.get(task.task_type)
        if not handler:
            await task.mark_failed(f"No handler for type {task.task_type}")
            return

        try:
            await task.mark_running()
            # Execute handler
            result = await handler(task)
            await task.mark_completed(result)
        except Exception as e:
            logger.exception(f"Task {task.id} failed")
            await task.mark_failed(str(e))
