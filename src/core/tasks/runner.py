"""
BackgroundTaskRunner - QThread with own asyncio event loop.

Provides non-blocking background task execution with Qt signal-based result delivery.
UI thread never blocks waiting for results.

Usage:
    runner = BackgroundTaskRunner()
    runner.task_completed.connect(on_result)
    runner.start()
    
    runner.submit("handler_name", task_id, *args)
"""
import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from loguru import logger

from PySide6.QtCore import QThread, Signal


@dataclass
class BackgroundTask:
    """Task to run in background thread."""
    task_id: str
    handler_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 0=HIGH, 1=NORMAL, 2=LOW


class BackgroundTaskRunner(QThread):
    """
    Background thread with own asyncio event loop.
    
    Executes tasks submitted via thread-safe queue and emits
    Qt signals on completion for thread-safe UI updates.
    
    Signals:
        task_started: Emitted when task begins execution
        task_completed: Emitted with result on success
        task_failed: Emitted with error message on failure
        task_progress: Emitted for progress updates (optional)
    """
    
    # Qt Signals for cross-thread communication
    task_started = Signal(str)                  # task_id
    task_completed = Signal(str, object)        # task_id, result
    task_failed = Signal(str, str)              # task_id, error_message
    task_progress = Signal(str, int, str)       # task_id, percent, message
    
    def __init__(self, task_system=None, parent=None):
        super().__init__(parent)
        
        # Reference to TaskSystem for handler lookup
        self._task_system = task_system
        
        # Thread-safe task queue (priority queue)
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # Control flags
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Shutdown coordination
        self._shutdown_event = threading.Event()
        
    def _get_handler(self, name: str) -> Optional[Callable]:
        """
        Get handler from TaskSystem.
        
        Args:
            name: Handler identifier
            
        Returns:
            Handler function or None
        """
        if self._task_system:
            return self._task_system._handlers.get(name)
        return None
        
    def submit(self, handler_name: str, task_id: str, *args, 
               priority: int = 1, **kwargs) -> str:
        """
        Submit a task for background execution.
        
        Non-blocking - returns immediately.
        Result delivered via task_completed signal.
        
        Args:
            handler_name: Name of registered handler
            task_id: Unique task identifier
            *args: Positional args for handler
            priority: 0=HIGH, 1=NORMAL, 2=LOW
            **kwargs: Keyword args for handler
            
        Returns:
            task_id for tracking
        """
        task = BackgroundTask(
            task_id=task_id,
            handler_name=handler_name,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        # Queue with priority (lower = higher priority)
        self._task_queue.put((priority, task))
        logger.debug(f"BackgroundTaskRunner: Queued task {task_id[:8]}... ({handler_name})")
        
        return task_id
    
    def run(self):
        """
        Thread entry point - runs own asyncio event loop.
        
        DO NOT CALL DIRECTLY - use start() instead.
        """
        logger.info("BackgroundTaskRunner: Starting background thread")
        self._running = True
        
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            # Run the worker until shutdown
            self._loop.run_until_complete(self._worker_loop())
        except Exception as e:
            logger.error(f"BackgroundTaskRunner: Event loop error: {e}")
        finally:
            # Cleanup
            self._loop.close()
            self._loop = None
            self._running = False
            logger.info("BackgroundTaskRunner: Thread stopped")
    
    async def _worker_loop(self):
        """Main worker loop - processes tasks from queue."""
        logger.debug("BackgroundTaskRunner: Worker loop started")
        
        while self._running:
            try:
                # Non-blocking get with timeout for responsive shutdown
                try:
                    priority, task = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if shutdown requested
                    if self._shutdown_event.is_set():
                        break
                    continue
                
                # Handle shutdown sentinel
                if task is None:
                    break
                    
                await self._execute_task(task)
                self._task_queue.task_done()
                
            except Exception as e:
                logger.error(f"BackgroundTaskRunner: Worker error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _execute_task(self, task: BackgroundTask):
        """Execute a single task and emit result signal."""
        handler = self._get_handler(task.handler_name)
        
        if not handler:
            error_msg = f"Handler '{task.handler_name}' not found"
            logger.error(f"BackgroundTaskRunner: {error_msg}")
            self.task_failed.emit(task.task_id, error_msg)
            return
        
        # Emit started signal
        self.task_started.emit(task.task_id)
        
        try:
            # Execute handler (async or sync)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*task.args, **task.kwargs)
            else:
                # Run sync handler in executor to not block
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: handler(*task.args, **task.kwargs)
                )
            
            # Emit success
            self.task_completed.emit(task.task_id, result)
            logger.debug(f"BackgroundTaskRunner: Task {task.task_id[:8]}... completed")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"BackgroundTaskRunner: Task {task.task_id[:8]}... failed: {e}")
            self.task_failed.emit(task.task_id, error_msg)
    
    def stop(self):
        """Request graceful shutdown."""
        logger.info("BackgroundTaskRunner: Shutdown requested")
        self._running = False
        self._shutdown_event.set()
        
        # Send shutdown sentinel
        self._task_queue.put((999, None))
        
        # Wait for thread to finish (with timeout)
        self.wait(5000)  # 5 second timeout
    
    def is_running(self) -> bool:
        """Check if runner is active."""
        return self._running and self.isRunning()
