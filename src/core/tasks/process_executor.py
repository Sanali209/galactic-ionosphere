"""
ProcessExecutor - ProcessPoolExecutor wrapper for CPU-heavy non-LLM tasks.

Provides async/await interface for running CPU-intensive work in separate processes,
bypassing Python's GIL for true parallelism.

Uses 'spawn' context for Windows compatibility and to avoid shared state issues.
"""
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Any, Optional
from loguru import logger


class ProcessExecutor:
    """
    Wrapper for ProcessPoolExecutor with asyncio integration.
    
    Features:
    - Async/await interface for non-blocking execution
    - Spawn context for Windows compatibility
    - Graceful shutdown support
    - Worker pool management
    
    Usage:
        executor = ProcessExecutor(max_workers=4)
        result = await executor.submit(cpu_heavy_function, arg1, arg2)
        executor.shutdown()
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize process pool.
        
        Args:
            max_workers: Number of worker processes
        """
        # Use 'spawn' context for Windows compatibility
        # This ensures clean process state and avoids shared memory issues
        ctx = mp.get_context('spawn')
        
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx
        )
        self._max_workers = max_workers
        
        logger.info(f"ProcessExecutor initialized with {max_workers} workers")
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit function to process pool and await result.
        
        Args:
            func: Function to execute (must be picklable - top-level function)
            *args: Positional arguments for func (must be picklable)
            **kwargs: Keyword arguments for func (must be picklable)
            
        Returns:
            Result from func execution
            
        Raises:
            Exception if func execution fails in worker process
            
        Note:
            - func must be importable (top-level function, not lambda/closure)
            - All args/kwargs must be picklable
            - No open file handles, no database connections
        """
        loop = asyncio.get_running_loop()
        
        # Submit to process pool and await result (non-blocking)
        result = await loop.run_in_executor(
            self._pool,
            func,
            *args,
            **kwargs
        )
        
        return result
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the process pool.
        
        Args:
            wait: If True, wait for all pending tasks to complete
        """
        logger.info(f"ProcessExecutor shutting down (wait={wait})...")
        self._pool.shutdown(wait=wait)
        logger.info("ProcessExecutor shutdown complete")
    
    @property
    def max_workers(self) -> int:
        """Get the number of worker processes."""
        return self._max_workers
