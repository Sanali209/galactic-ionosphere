"""
Engine Proxy Implementation.

Acts as the bridge (Client) between the Main UI Thread and the Background Engine Thread.
Communicates via Thread-safe Queues and Qt Signals.
"""
import asyncio
from typing import Any, Callable, Coroutine, Optional
from PySide6.QtCore import QObject, Signal
from loguru import logger

from .thread import EngineThread


class EngineProxy(QObject):
    """
    Proxy service living in the Main Thread.
    Manages the Background Engine Thread.
    
    Note: Does not inherit from BaseSystem to avoid metaclass conflicts with QObject.
    Manually implements BaseSystem interface (initialize, shutdown, depends_on).
    
    Usage:
        proxy = locator.get_system(EngineProxy)
        await proxy.start()
        
        # Run task on engine
        future = proxy.submit(some_coroutine())
    """
    
    # BaseSystem interface
    depends_on = [] 
    
    # Signals (relayed from EngineThread)
    engine_started = Signal()
    engine_stopped = Signal()
    engine_error = Signal(str)
    
    # Generic status relay (category, status, message)
    status_update = Signal(str, str, str)
    
    # Task System Signals (Relayed from Engine)
    task_started = Signal(str)
    task_completed = Signal(str, object)
    task_failed = Signal(str, str)
    task_progress = Signal(str, int, str)
    
    def __init__(self, locator: Any, config: Any):
        super().__init__()  # Initialize QObject
        
        # Manually set BaseSystem attributes
        self.locator = locator
        self.config = config
        self._initialized = False
        
        self.thread: Optional[EngineThread] = None
        self._bootstrap_func: Optional[Callable] = None
        self._ready_event: Optional[asyncio.Event] = None
        self._engine_ready = False
        
    def set_bootstrap(self, func: Callable):
        """Set the bootstrap function for the engine locator."""
        self._bootstrap_func = func

    async def initialize(self):
        """Initialize proxy (does not start thread yet)."""
        logger.info("EngineProxy initialized")
        self._ready_event = asyncio.Event()
        self.thread = EngineThread("UExplorerEngine")
        
        # Pass correct config path to Engine (fix specific config loading)
        if hasattr(self.config, 'filepath'):
            self.thread._config_path = self.config.filepath
        else:
            logger.warning("EngineProxy: Config has no filepath, Engine using default config.json")
        
        # Connect signals
        self.thread.started_signal.connect(self._on_started)
        self.thread.stopped_signal.connect(self._on_stopped)
        self.thread.error_signal.connect(self._on_error)
        
        # Connect Task Signals
        self.thread.task_started.connect(self.task_started.emit)
        self.thread.task_completed.connect(self.task_completed.emit)
        self.thread.task_failed.connect(self.task_failed.emit)
        self.thread.task_progress.connect(self.task_progress.emit)
        
        self._is_ready = True

    # ... (shutdown, start_engine unchanged) ...

    def submit_task(self, handler_name: str, *args, priority: int = 1, **kwargs) -> str:
        """
        Submit a background task via the Engine.
        Blocking call (waits for task_id generation).
        
        Args:
            handler_name: Name of registered handler
            args: Arguments for handler
            priority: 0=High, 1=Normal, 2=Low
            kwargs: Keyword arguments
            
        Returns:
            task_id (str)
        """
        if not self.thread or not self.thread.isRunning():
            raise RuntimeError("Engine not running")
        
        # Define helper task
        async def _submit_job():
            try:
                from PySide6.QtCore import QThread
                thread = QThread.currentThread()
                if hasattr(thread, 'locator') and thread.locator:
                    sl = thread.locator
                else:
                    from src.core.locator import get_active_locator
                    sl = get_active_locator()
                
                from src.core.tasks.system import TaskSystem
                ts = sl.get_system(TaskSystem)
                return await ts.submit(handler_name, *args, priority=priority, **kwargs)
            except Exception as e:
                logger.error(f"EngineProxy submit_task failed: {e}")
                raise
            
        future = self.thread.submit(_submit_job())
        return future.result()  # Block and wait for ID generation (fast)
    
    def _on_started(self):
        """Handle Engine thread started signal."""
        logger.info("Engine thread started - marking as ready")
        self._engine_ready = True
        if self._ready_event:
            # Set event in main thread's event loop
            asyncio.get_event_loop().call_soon_threadsafe(self._ready_event.set)
    
    def _on_stopped(self):
        """Handle Engine thread stopped signal."""
        logger.info("Engine thread stopped")
        self._engine_ready = False
    
    def _on_error(self, error_msg: str):
        """Handle Engine thread error signal."""
        logger.error(f"Engine thread error: {error_msg}")
    
    def get_task_stats(self) -> dict:
        """
        Query task stats from Engine (Blocking).
        Returns immediately with cached values or empty dict if unavailable.
        """
        if not self.thread or not self.thread.isRunning():
            return {}
            
        async def _stats_job():
            try:
                from PySide6.QtCore import QThread
                thread = QThread.currentThread()
                if hasattr(thread, 'locator') and thread.locator:
                    sl = thread.locator
                else:
                    from src.core.locator import get_active_locator
                    sl = get_active_locator()
                    
                from src.core.tasks.system import TaskSystem
                ts = sl.get_system(TaskSystem)
                return {
                    "workers": len(ts._workers),
                    "queue": ts._queue.qsize(),
                }
            except Exception:
                return {}
            
        try:
            future = self.thread.submit(_stats_job())
            return future.result(timeout=1.0)  # Block and wait for result
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {}
        
    async def shutdown(self):
        """Request engine shutdown."""
        if self.thread and self.thread.isRunning():
            logger.info("Requesting engine shutdown...")
            self.thread.stop()
            
            # Wait up to 30s for graceful shutdown (Engine services need time)
            logger.info("Waiting for engine thread to stop (max 30s)...")
            self.thread.wait(30000) # 30 seconds
            
            if self.thread.isRunning():
                logger.error("Engine thread did not stop gracefully after 30s, terminating forcefully")
                self.thread.terminate()
                self.thread.wait(2000)
            else:
                logger.info("Engine thread stopped gracefully")
        
        self._engine_ready = False

    async def start_processing(self):
        """
        Signal TaskSystem to start workers.
        Should be called after critical systems (like AI models) are preloaded.
        """
        if not self.thread or not self.thread.isRunning():
            logger.warning("Cannot start processing: Engine not running")
            return 0
            
        async def _start_workers():
            try:
                from PySide6.QtCore import QThread
                thread = QThread.currentThread()
                if hasattr(thread, 'locator') and thread.locator:
                    sl = thread.locator
                else:
                    from src.core.locator import get_active_locator
                    sl = get_active_locator()
                    
                from src.core.tasks.system import TaskSystem
                ts = sl.get_system(TaskSystem)
                count = await ts.start_workers()
                return count
            except Exception as e:
                logger.error(f"EngineProxy start_workers failed: {e}")
                return 0
            
        future = self.thread.submit(_start_workers())
        # Wait for result from Engine Thread
        result = future.result()
        return result
        self._engine_ready = False
        if self._ready_event:
            self._ready_event.clear()

    async def start_engine(self):
        """Start the background engine thread."""
        if not self.thread:
            raise RuntimeError("EngineProxy not initialized")
            
        if self.thread.isRunning():
            logger.warning("Engine thread already running")
            return
            
        if not self._bootstrap_func:
            logger.warning("No bootstrap function set for Engine! Engine will be empty.")
            
        self._ready_event.clear()
        self._engine_ready = False
        self.thread.configure(self._bootstrap_func, self.config.config_path if hasattr(self.config, 'config_path') else "config.json")
        self.thread.start()
        
        logger.info("Engine thread start requested")
    
    async def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for Engine to be ready.
        
        Args:
            timeout: Max seconds to wait
            
        Returns:
            True if ready, False if timeout
        """
        if self._engine_ready:
            return True
        
        if not self._ready_event:
            return False
            
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Engine not ready after {timeout}s")
            return False

    def submit(self, coro: Coroutine) -> asyncio.Future:
        """Submit coroutine to Engine loop."""
        if not self.thread or not self.thread.isRunning():
            raise RuntimeError("Engine not running")
        return self.thread.submit(coro)

    # --- Signal Handlers (Main Thread) ---
    
    def _on_started(self):
        logger.info("Engine is running")
        self._engine_ready = True
        if self._ready_event:
            self._ready_event.set()
        self.engine_started.emit()
        
    def _on_stopped(self):
        logger.info("Engine stopped")
        self.engine_stopped.emit()
        
    def _on_error(self, msg):
        logger.error(f"Engine reported error: {msg}")
        self.engine_error.emit(msg)
