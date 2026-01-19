"""
AIExtractor - Base class for AI-powered extractors.

Provides shared functionality for extractors that use LLMWorkerService:
- _get_llm_service() - Get LLM worker if available
- _get_ai_executor() - Get shared ThreadPoolExecutor
- Common try-worker/fallback-legacy pattern
"""
from abc import abstractmethod
from typing import List, Dict, Any, Optional
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord


class AIExtractor(Extractor):
    """
    Base class for extractors that use LLMWorkerService.
    
    Subclasses must implement:
    - _extract_via_worker(files, llm_service) - Using LLM worker
    - _extract_legacy(files) - Fallback without worker
    
    The base extract() method handles the try-worker/fallback pattern.
    """
    
    def _get_llm_service(self):
        """
        Get LLMWorkerService from locator if available.
        
        Returns:
            LLMWorkerService instance or None
        """
        if not self.locator:
            return None
        try:
            from src.core.llm.worker_service import LLMWorkerService
            service = self.locator.get_system(LLMWorkerService)
            if service and service.is_available():
                return service
        except (KeyError, ImportError):
            pass
        return None
    
    def _get_ai_executor(self):
        """
        Get shared AI ThreadPoolExecutor from TaskSystem.
        
        Returns:
            ThreadPoolExecutor or None
        """
        if not self.locator:
            return None
        try:
            from src.core.tasks.system import TaskSystem
            task_system = self.locator.get_system(TaskSystem)
            return task_system.get_ai_executor() if task_system else None
        except (KeyError, ImportError):
            return None
    
    async def _run_in_ai_executor(self, func, *args):
        """
        Run a sync function in the shared AI executor.
        
        Standardized method for running AI inference in thread pool.
        Falls back to asyncio.to_thread if no executor available.
        
        Args:
            func: Sync function to run
            *args: Arguments to pass to func
            
        Returns:
            Result from func
        """
        import asyncio
        executor = self._get_ai_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, func, *args)
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Extract with LLMWorker, fallback to legacy.
        
        Args:
            files: List of FileRecord to process
            
        Returns:
            Dict mapping file_id to extraction result
        """
        if not files:
            return {}
        
        valid_files = [f for f in files if self.can_process(f)]
        if not valid_files:
            return {f._id: None for f in files}
        
        # Try LLMWorkerService first
        llm_service = self._get_llm_service()
        if llm_service:
            try:
                logger.debug(f"[{self.name}] Using LLMWorkerService for {len(valid_files)} files")
                return await self._extract_via_worker(valid_files, llm_service)
            except Exception as e:
                logger.warning(f"[{self.name}] Worker failed, falling back to legacy: {e}")
        
        # Fallback to legacy
        logger.debug(f"[{self.name}] Using legacy extraction for {len(valid_files)} files")
        return await self._extract_legacy(valid_files)
    
    @abstractmethod
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """
        Extract using LLMWorkerService.
        
        Args:
            files: Pre-filtered files that can be processed
            llm_service: LLMWorkerService instance
            
        Returns:
            Dict mapping file_id to extraction result
        """
        ...
    
    @abstractmethod
    async def _extract_legacy(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Legacy extraction without LLM worker.
        
        Args:
            files: Pre-filtered files that can be processed
            
        Returns:
            Dict mapping file_id to extraction result
        """
        ...
