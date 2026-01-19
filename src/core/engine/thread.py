"""
Engine Thread Implementation.

Provides a dedicated QThread running a separate asyncio event loop.
This is the "Server" component of the Client-Server internal architecture.
"""
import asyncio
import threading
from typing import Optional, Any, Callable, Coroutine
from PySide6.QtCore import QThread, Signal, QObject
from loguru import logger

class EngineThread(QThread):
    """
    Dedicated background thread for heavy processing engine.
    
    Runs a persistent asyncio event loop independent of the main UI loop.
    Manages its own ServiceLocator and System lifecycle.
    """
    
    # Signals for Main Thread communication
    started_signal = Signal()
    stopped_signal = Signal()
    error_signal = Signal(str)
    
    # Task System Signals (Relayed)
    task_started = Signal(str)        # task_id
    task_completed = Signal(str, object) # task_id, result
    task_failed = Signal(str, str)    # task_id, error
    task_progress = Signal(str, int, str) # task_id, percent, status
    
    def __init__(self, name: str = "Engine"):
        super().__init__()
        self.setObjectName(name)
        
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        
        # The Engine's ServiceLocator (initialized within the thread)
        self.locator: Any = None
        
        # Configuration pass-through
        self._bootstrap_func: Optional[Callable[[Any], Coroutine]] = None
        self._config_path: str = "config.json"
        
    def configure(self, bootstrap_func: Callable, config_path: str = "config.json"):
        """
        Configure the engine before starting.
        
        Args:
            bootstrap_func: Async function(thread) -> ServiceLocator
                            Called inside the thread to build systems.
            config_path: Path to config file
        """
        self._bootstrap_func = bootstrap_func
        self._config_path = config_path
        
    def run(self):
        """Thread entry point."""
        logger.info(f"Engine thread '{self.objectName()}' starting...")
        
        # 1. Create and Set Event Loop
        try:
            # On Windows/Proactor, we must be careful. 
            # new_event_loop() creates the correct loop for the platform.
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # 2. Run Initialization
            if self._bootstrap_func:
                logger.debug("Running engine bootstrap...")
                # Pass SELF (EngineThread) to bootstrap so it can connect signals
                self.locator = self.loop.run_until_complete(self._bootstrap_func(self))
            
            self._shutdown_event = asyncio.Event()
            
            # Notify Main Thread we are ready
            self.started_signal.emit()
            logger.info("Engine loop running")
            
            # 3. Run Forever
            self.loop.run_forever()
            
        except Exception as e:
            logger.critical(f"Engine thread crashed: {e}")
            self.error_signal.emit(str(e))
            
        finally:
            # 4. Cleanup
            logger.info("Engine thread shutting down...")
            if self.loop:
                # Run shutdown routines
                try:
                    if self.locator:
                        # 1. Stop TaskSystem FIRST (stop processing new tasks)
                        try:
                            from src.core.tasks.system import TaskSystem
                            if self.locator.has_system(TaskSystem):
                                logger.info("Stopping Engine TaskSystem first...")
                                ts = self.locator.get_system(TaskSystem)
                                self.loop.run_until_complete(ts.shutdown())
                        except Exception as e:
                            logger.error(f"TaskSystem manual shutdown failed: {e}")
                            
                        # 2. Stop everything else
                        if hasattr(self.locator, 'stop_all'):
                             self.loop.run_until_complete(self.locator.stop_all())
                except Exception as e:
                    logger.error(f"Engine shutdown error: {e}")
                    
                # Close loop
                try:
                    tasks = asyncio.all_tasks(self.loop)
                    for task in tasks:
                        task.cancel()
                    
                    # Allow cancellation to propagate
                    if tasks:
                        self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                        
                    self.loop.close()
                except Exception as e:
                    logger.error(f"Engine loop close error: {e}")
            
            self.stopped_signal.emit()
            logger.info("Engine thread stopped")

    def stop(self):
        """Request engine stop (Thread-safe)."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            
    def submit(self, coro: Coroutine) -> asyncio.Future:
        """
        Submit a coroutine to the engine loop.
        
        Returns:
            concurrent.futures.Future (Thread-safe)
        """
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Engine loop is not running")
            
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
        
    def submit_sync(self, coro: Coroutine, timeout: float = None) -> Any:
        """
        Submit coroutine and wait for result (Blocking).
        """
        future = self.submit(coro)
        return future.result(timeout)
