"""
LLM Worker Service

Manages a pool of worker processes for non-blocking LLM inference.
Provides async API for submitting jobs and awaiting results.
"""
import asyncio
import multiprocessing as mp
from multiprocessing import Queue
from typing import Dict, Optional, Any
from concurrent.futures import Future
import threading
from loguru import logger

from src.core.base_system import BaseSystem
from src.core.llm.models import (
    LLMJobRequest, 
    LLMJobResult, 
    JobPriority,
    JobStatus,
    SHUTDOWN_SENTINEL
)


class LLMWorkerService(BaseSystem):
    """
    Non-blocking LLM inference service using multiprocessing workers.
    
    Features:
    - Separate worker processes for GPU isolation
    - Priority queue for job ordering
    - Async Future-based API for non-blocking await
    - Qt signal integration for UI updates
    
    Usage:
        service = locator.get_system(LLMWorkerService)
        future = await service.submit_job(LLMJobRequest(
            task_type="clip",
            file_paths=["/path/to/image.jpg"]
        ))
        result = await future  # Non-blocking
    """
    
    depends_on = ["ConfigManager"]
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        
        self._request_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._workers: list = []
        self._pending_jobs: Dict[str, asyncio.Future] = {}
        self._result_poll_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Configuration - Single worker only (resource constraints)
        self._num_workers = 1  # User requirement: only 1 LLM worker
        self._queue_max_size = 100
        
    async def initialize(self):
        """Initialize worker pool and queues."""
        logger.info("LLMWorkerService initializing...")
        
        # Check if enabled in config
        # TEMPORARY FIX: Force enable until config parsing issue resolved
        enabled = True
        logger.warning("⚠️ LLM Worker FORCE ENABLED (bypassing config until parsing fixed)")
        
        # Still log config for debugging
        try:
            # DIAGNOSTIC: Log config object details
            logger.info(f"🔍 CONFIG DEBUG - Config object type: {type(self.config)}")
            logger.info(f"🔍 CONFIG DEBUG - Has data attr: {hasattr(self.config, 'data')}")
            
            if hasattr(self.config, 'data'):
                logger.info(f"🔍 CONFIG DEBUG - Config.data type: {type(self.config.data)}")
                
                # Dump full config for inspection
                if hasattr(self.config.data, 'model_dump'):
                    full_config = self.config.data.model_dump()
                    logger.info(f"🔍 CONFIG DEBUG - Full config keys: {list(full_config.keys())}")
                    if 'llm_workers' in full_config:
                        logger.info(f"🔍 CONFIG DEBUG - llm_workers section: {full_config['llm_workers']}")
                
                logger.info(f"🔍 CONFIG DEBUG - Has llm_workers attr: {hasattr(self.config.data, 'llm_workers')}")
                
                if hasattr(self.config.data, 'llm_workers'):
                    llm_workers = self.config.data.llm_workers
                    logger.info(f"🔍 CONFIG DEBUG - llm_workers object: {llm_workers}")
                    logger.info(f"🔍 CONFIG DEBUG - llm_workers type: {type(llm_workers)}")
                    enabled = llm_workers.enabled
                    logger.info(f"🔍 CONFIG DEBUG - llm_workers.enabled = {enabled} (type: {type(enabled)})")
                else:
                    logger.warning("🔍 CONFIG DEBUG - llm_workers attribute NOT FOUND in config.data")
            else:
                logger.warning("🔍 CONFIG DEBUG - config.data attribute NOT FOUND")
                
            logger.debug(f"llm_workers.enabled = {enabled}")
        except Exception as e:
            logger.error(f"Failed to read llm_workers.enabled: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        if not enabled:

            logger.warning("LLMWorkerService DISABLED - extractors will use legacy ThreadPoolExecutor")
            await super().initialize()
            return
        
        # Build model configs
        model_configs = self._build_model_configs()
        
        # Use spawn context for Windows compatibility
        ctx = mp.get_context('spawn')
        self._request_queue = ctx.Queue(maxsize=self._queue_max_size)
        self._result_queue = ctx.Queue(maxsize=self._queue_max_size)
        
        # Start single worker process
        worker = ctx.Process(
            target=_worker_process_main,
            args=(0, self._request_queue, self._result_queue, model_configs),
            name="LLMWorker-0"
        )
        worker.start()
        self._workers.append(worker)
        logger.info("LLMWorker-0 process started, waiting for ready signal...")
        
        # Wait for worker to signal ready (models loaded)
        await self._wait_for_ready()
        
        self._running = True
        
        # Start result polling task
        self._result_poll_task = asyncio.create_task(self._poll_results())
        
        logger.info("LLMWorkerService initialized with 1 worker")
        
        await super().initialize()
    
    async def shutdown(self):
        """Shutdown workers gracefully."""
        logger.info("LLMWorkerService shutting down...")
        self._running = False
        
        # Cancel result polling
        if self._result_poll_task and not self._result_poll_task.done():
            self._result_poll_task.cancel()
            try:
                await self._result_poll_task
            except asyncio.CancelledError:
                pass
        
        # Send shutdown sentinel to each worker
        if self._request_queue:
            for _ in self._workers:
                try:
                    self._request_queue.put_nowait(SHUTDOWN_SENTINEL)
                except Exception:
                    pass
        
        # Wait for workers to terminate
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.name}")
                worker.terminate()
        
        # Cancel pending futures
        for job_id, future in self._pending_jobs.items():
            if not future.done():
                future.set_exception(RuntimeError("Service shutdown"))
        
        self._pending_jobs.clear()
        self._workers.clear()
        
        await super().shutdown()
        logger.info("LLMWorkerService shutdown complete")
    
    async def _wait_for_ready(self, timeout: float = 60.0):
        """
        Wait for worker process to signal ready.
        
        Worker sends __READY__ result after loading models.
        Non-blocking wait with timeout.
        
        Args:
            timeout: Max seconds to wait for ready signal
            
        Raises:
            TimeoutError: If worker doesn't signal ready within timeout
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Non-blocking check for ready signal
                if not self._result_queue.empty():
                    result = self._result_queue.get_nowait()
                    if result.job_id == "__READY__":
                        logger.info("LLMWorker ready signal received")
                        return
                # Brief sleep to avoid busy-waiting
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Waiting for ready: {e}")
                await asyncio.sleep(0.1)
        
        raise TimeoutError(f"LLMWorker did not signal ready within {timeout}s")
    
    def _build_model_configs(self) -> Dict[str, Any]:
        """Build model configuration dict from app config."""
        configs = {}
        try:
            if hasattr(self.config, 'data'):
                data = self.config.data
                
                # Processing config for detection models
                if hasattr(data, 'processing'):
                    proc = data.processing
                    if hasattr(proc, 'detection'):
                        det = proc.detection
                        if hasattr(det, 'yolo'):
                            configs['yolo'] = det.yolo.model_dump() if hasattr(det.yolo, 'model_dump') else {}
                        if hasattr(det, 'grounding_dino'):
                            configs['grounding_dino'] = det.grounding_dino.model_dump() if hasattr(det.grounding_dino, 'model_dump') else {}
                    
                    if hasattr(proc, 'wd_tagger'):
                        configs['wdtagger'] = proc.wd_tagger.model_dump() if hasattr(proc.wd_tagger, 'model_dump') else {}
                
                # LLM worker specific configs
                if hasattr(data, 'llm_workers') and hasattr(data.llm_workers, 'models'):
                    for model_name, model_config in data.llm_workers.models.items():
                        if model_name not in configs:
                            configs[model_name] = model_config if isinstance(model_config, dict) else {}
        except Exception as e:
            logger.debug(f"Error building model configs: {e}")
        
        return configs
    
    async def submit_job(self, request: LLMJobRequest) -> asyncio.Future:
        """
        Submit a job for processing.
        
        Args:
            request: LLMJobRequest with task_type and file_paths
            
        Returns:
            asyncio.Future that resolves to LLMJobResult
        """
        if not self._running:
            raise RuntimeError("LLMWorkerService not running")
        
        # Create future for result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_jobs[request.job_id] = future
        
        # Put in queue (non-blocking with timeout)
        try:
            self._request_queue.put(request, timeout=5.0)
            logger.debug(f"Submitted LLM job {request.job_id[:8]}... ({request.task_type})")
        except Exception as e:
            del self._pending_jobs[request.job_id]
            future.set_exception(RuntimeError(f"Queue full: {e}"))
        
        return future
    
    async def _poll_results(self):
        """Poll result queue and resolve pending futures."""
        while self._running:
            try:
                # Non-blocking check
                while not self._result_queue.empty():
                    try:
                        result: LLMJobResult = self._result_queue.get_nowait()
                        future = self._pending_jobs.pop(result.job_id, None)
                        if future and not future.done():
                            future.set_result(result)
                            logger.debug(f"Resolved job {result.job_id[:8]}... success={result.success}")
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")
                
                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.05)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Result poll error: {e}")
                await asyncio.sleep(0.1)
    
    def get_queue_depth(self) -> int:
        """Get approximate number of pending jobs."""
        try:
            return self._request_queue.qsize() + len(self._pending_jobs)
        except Exception:
            return len(self._pending_jobs)
    
    def is_available(self) -> bool:
        """Check if service is ready to accept jobs."""
        return self._running and len(self._workers) > 0


def _worker_process_main(
    worker_id: int,
    request_queue: Queue,
    result_queue: Queue,
    model_configs: Dict[str, Any]
):
    """
    Worker process main function.
    
    Runs in a separate process, isolated from the main UI process.
    Pre-loads ALL enabled models at startup (no load/unload later).
    """
    import time
    from loguru import logger
    
    # Configure logger for this process
    logger.info(f"[Worker-{worker_id}] Starting, loading models...")
    
    # Pre-load ALL enabled models at startup (user requirement: no load/unload)
    models = {}
    for model_name in ['clip', 'blip', 'wdtagger', 'yolo', 'grounding_dino']:
        if model_name in model_configs:
            try:
                if model_name == 'clip':
                    _ensure_clip_model(models, model_configs.get(model_name, {}))
                elif model_name == 'blip':
                    _ensure_blip_model(models, model_configs.get(model_name, {}))
                # Add other models as needed
                logger.info(f"[Worker-{worker_id}] ✓ Loaded {model_name}")
            except Exception as e:
                logger.warning(f"[Worker-{worker_id}] ✗ {model_name}: {e}")
    
    # Signal ready
    result_queue.put(LLMJobResult(
        job_id="__READY__", 
        success=True, 
        task_type="__READY__"
    ))
    logger.info(f"[Worker-{worker_id}] Ready, models in memory")
    
    while True:
        try:
            request: LLMJobRequest = request_queue.get()
            
            # Check for shutdown
            if request.task_type == "__SHUTDOWN__":
                logger.info(f"[Worker-{worker_id}] Received shutdown signal")
                break
            
            start_time = time.time()
            
            try:
                # Process based on task type
                result_data = _process_request(
                    request, 
                    models, 
                    model_configs.get(request.task_type, {})
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                result = LLMJobResult(
                    job_id=request.job_id,
                    success=True,
                    task_type=request.task_type,
                    data=result_data,
                    elapsed_ms=elapsed_ms
                )
                
            except Exception as e:
                import traceback
                logger.error(f"[Worker-{worker_id}] Job {request.job_id[:8]} failed: {e}\n{traceback.format_exc()}")
                result = LLMJobResult.failure(
                    job_id=request.job_id,
                    task_type=request.task_type,
                    error=str(e)
                )
            
            result_queue.put(result)
            
        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Fatal error: {e}")
            break
    
    logger.info(f"[Worker-{worker_id}] Exiting")


def _process_request(
    request: LLMJobRequest,
    models: Dict[str, Any],
    config: Dict[str, Any]
) -> Any:
    """
    Process a single request with the appropriate model.
    
    Args:
        request: The job request
        models: Dict of loaded models (modified in-place)
        config: Model configuration
        
    Returns:
        Result data (format depends on task_type)
    """
    task_type = request.task_type
    file_paths = request.file_paths
    options = request.options
    
    if task_type == "clip":
        return _run_clip(file_paths, models, config, options)
    elif task_type == "blip":
        return _run_blip(file_paths, models, config, options)
    elif task_type == "wdtagger":
        return _run_wdtagger(file_paths, models, config, options)
    elif task_type == "yolo":
        return _run_yolo(file_paths, models, config, options)
    elif task_type == "grounding_dino":
        return _run_grounding_dino(file_paths, models, config, options)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ==================== Model Inference Functions ====================

def _ensure_clip_model(models: dict, config: dict):
    """Ensure CLIP model is loaded."""
    if "clip" in models:
        return models["clip"]
    
    import torch
    import clip
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.get("model_name", "ViT-B/32")
    model, preprocess = clip.load(model_name, device=device)
    
    models["clip"] = {
        "model": model,
        "preprocess": preprocess,
        "device": device
    }
    logger.info(f"[Worker] Loaded CLIP model on {device}")
    return models["clip"]


def _run_clip(file_paths: list, models: dict, config: dict, options: dict) -> dict:
    """Run CLIP embedding extraction."""
    import torch
    from PIL import Image
    
    clip_model = _ensure_clip_model(models, config)
    model = clip_model["model"]
    preprocess = clip_model["preprocess"]
    device = clip_model["device"]
    
    results = {}
    
    for file_path in file_paths:
        try:
            image = Image.open(file_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                embedding = features.cpu().numpy().flatten().tolist()
            
            results[file_path] = {
                "vector": embedding,
                "dimension": len(embedding)
            }
        except Exception as e:
            logger.error(f"CLIP failed for {file_path}: {e}")
            results[file_path] = None
    
    return results


def _ensure_blip_model(models: dict, config: dict):
    """Ensure BLIP model is loaded."""
    if "blip" in models:
        return models["blip"]
    
    import os
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")
    model_name = config.get("model_name", "Salesforce/blip-image-captioning-base")
    
    processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained(model_name, token=hf_token).to(device)
    
    models["blip"] = {
        "model": model,
        "processor": processor,
        "device": device
    }
    logger.info(f"[Worker] Loaded BLIP model on {device}")
    return models["blip"]


def _run_blip(file_paths: list, models: dict, config: dict, options: dict) -> dict:
    """Run BLIP captioning."""
    import torch
    from PIL import Image
    
    blip_model = _ensure_blip_model(models, config)
    model = blip_model["model"]
    processor = blip_model["processor"]
    device = blip_model["device"]
    
    results = {}
    
    for file_path in file_paths:
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_length=50)
                caption = processor.decode(output[0], skip_special_tokens=True)
            
            results[file_path] = {"caption": caption}
        except Exception as e:
            logger.error(f"BLIP failed for {file_path}: {e}")
            results[file_path] = None
    
    return results


def _run_wdtagger(file_paths: list, models: dict, config: dict, options: dict) -> dict:
    """Run WD Tagger for auto-tagging."""
    # WDTagger requires specific model loading - defer to service for now
    # This is a placeholder - actual implementation will require WD model setup
    logger.warning("WDTagger worker inference not yet implemented")
    return {fp: None for fp in file_paths}


def _run_yolo(file_paths: list, models: dict, config: dict, options: dict) -> dict:
    """Run YOLO object detection."""
    # YOLO uses ultralytics - defer to service for now
    logger.warning("YOLO worker inference not yet implemented")
    return {fp: None for fp in file_paths}


def _run_grounding_dino(file_paths: list, models: dict, config: dict, options: dict) -> dict:
    """Run GroundingDINO detection."""
    # GroundingDINO requires specific setup - defer to service for now
    logger.warning("GroundingDINO worker inference not yet implemented")
    return {fp: None for fp in file_paths}
