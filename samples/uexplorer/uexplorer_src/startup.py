"""
StartupOrchestrator - Flexible model preloading system.

Supports configurable model preloading with registry pattern.
Add any model by registering a preloader function.
"""
import asyncio
from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    from uexplorer_src.ui.dialogs.loading_dialog import LoadingDialog
    from src.core.service_locator import ServiceLocator


@dataclass
class ModelPreloader:
    """Definition of a model preloader."""
    id: str           # Unique identifier (e.g., "clip")
    name: str         # Display name (e.g., "CLIP Model")
    loader: Callable  # Async function that returns bool
    enabled: bool = True
    required: bool = False  # If True, startup fails if this fails


class StartupOrchestrator:
    """
    Orchestrates application startup with configurable model preloading.
    
    Usage:
        orchestrator = StartupOrchestrator(dialog)
        orchestrator.register_model("clip", "CLIP Model", clip_loader)
        orchestrator.register_model("blip", "BLIP Model", blip_loader)
        await orchestrator.run_preload(locator)
    """
    
    def __init__(self, dialog: "LoadingDialog"):
        self.dialog = dialog
        self.locator: Optional["ServiceLocator"] = None
        self._preloaders: Dict[str, ModelPreloader] = {}
        
        # Register default models
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default model preloaders."""
        self.register_model("clip", "CLIP Model", self._load_clip)
        self.register_model("blip", "BLIP Model", self._load_blip)
        self.register_model("wd_tagger", "WD-Tagger Model", self._load_wd_tagger)
        self.register_model("yolo", "YOLO Detector", self._load_yolo)
        self.register_model("grounding_dino", "GroundingDINO", self._load_grounding_dino)
    
    def register_model(self, id: str, name: str, loader: Callable, 
                       enabled: bool = True, required: bool = False):
        """
        Register a model preloader.
        
        Args:
            id: Unique identifier
            name: Display name
            loader: Async function(locator) -> bool
            enabled: Whether to preload by default
            required: If True, startup fails if preload fails
        """
        self._preloaders[id] = ModelPreloader(
            id=id,
            name=name,
            loader=loader,
            enabled=enabled,
            required=required
        )
    
    def set_enabled(self, model_id: str, enabled: bool):
        """Enable or disable a model preloader."""
        if model_id in self._preloaders:
            self._preloaders[model_id].enabled = enabled
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model IDs."""
        return [p.id for p in self._preloaders.values() if p.enabled]
    
    async def run_preload(self, locator: "ServiceLocator") -> bool:
        """
        Run all enabled model preloaders.
        
        Args:
            locator: Initialized ServiceLocator
            
        Returns:
            True if all required models loaded successfully
        """
        self.locator = locator
        success = True
        
        # Wait for Engine to be ready before submitting tasks
        try:
            from src.core.engine.proxy import EngineProxy
            proxy = locator.get_system(EngineProxy)
            
            logger.info("Waiting for Engine to be ready...")
            ready = await proxy.wait_for_ready(timeout=30.0)  # Increased from 15s to allow for GroundingDINO model loading
            
            if not ready:
                logger.error("Engine not ready - skipping model preloading")
                return False
                
            logger.info("✅ Engine is ready - starting model preloading")
            
        except Exception as e:
            logger.error(f"Failed to wait for Engine: {e}")
            return False
        
        # Filter enabled preloaders
        enabled = [p for p in self._preloaders.values() if p.enabled]
        
        for preloader in enabled:
            self.dialog.set_stage(preloader.id, "loading")
            
            try:
                result = await preloader.loader(locator)
                
                if result:
                    self.dialog.set_stage(preloader.id, "done")
                    logger.info(f"{preloader.name} preloaded")
                else:
                    self.dialog.set_stage(preloader.id, "skip")
                    logger.warning(f"{preloader.name} not available")
                    
                    if preloader.required:
                        success = False
                        
            except Exception as e:
                logger.error(f"{preloader.name} preload failed: {e}")
                self.dialog.set_stage(preloader.id, "error")
                
                if preloader.required:
                    success = False
        
        return success
    
    # ==================== Model Loaders ====================
    
    async def _load_clip(self, locator: "ServiceLocator") -> bool:
        """Load CLIP model on Engine."""
        try:
            from src.core.engine.proxy import EngineProxy
            proxy = locator.get_system(EngineProxy)
            
            # Submit task to Engine
            task = _ensure_engine_system("CLIPExtractor", "src.ucorefs.extractors.clip_extractor")
            future = proxy.submit(task)  # Returns concurrent.futures.Future
            import asyncio
            return await asyncio.wrap_future(future)
            
        except Exception as e:
            logger.warning(f"CLIP preload request failed: {e}")
            return False
    
    async def _load_blip(self, locator: "ServiceLocator") -> bool:
        """Load BLIP model on Engine."""
        try:
            from src.core.engine.proxy import EngineProxy
            proxy = locator.get_system(EngineProxy)
            
            task = _ensure_engine_system("BLIPExtractor", "src.ucorefs.extractors.blip_extractor")
            future = proxy.submit(task)  # Returns concurrent.futures.Future
            import asyncio
            return await asyncio.wrap_future(future)
            
        except Exception as e:
            logger.warning(f"BLIP preload request failed: {e}")
            return False
            
    async def _load_grounding_dino(self, locator: "ServiceLocator") -> bool:
        """Load GroundingDINO model on Engine."""
        try:
            from src.core.engine.proxy import EngineProxy
            proxy = locator.get_system(EngineProxy)
            
            task = _ensure_engine_system("GroundingDINOExtractor", "src.ucorefs.extractors.grounding_dino_extractor")
            future = proxy.submit(task)  # Returns concurrent.futures.Future
            import asyncio
            return await asyncio.wrap_future(future)
            
        except Exception as e:
            logger.warning(f"GroundingDINO preload request failed: {e}")
            return False

    async def _load_wd_tagger(self, locator: "ServiceLocator") -> bool:
        # For WD Tagger, it's a Service and already initialized in Engine
        return True

    async def _load_yolo(self, locator: "ServiceLocator") -> bool:
        """Check if YOLO backend is available."""
        try:
            from src.core.engine.proxy import EngineProxy
            from src.core.config import ConfigManager
            
            # Get proxy
            proxy = locator.get_system(EngineProxy)
            
            # Simple check: Is config enabled?
            try:
                config_sys = locator.get_system(ConfigManager)
                config = config_sys
            except Exception:
                # Fallback if ConfigManager not registered provided locator
                logger.warning("Startup: ConfigManager not found, assuming defaults")
                return True
                
            if not config.data.processing.detection.enabled:
                return False
                
            if not config.data.processing.detection.yolo.enabled:
                return False
            
            # Actually check if backend is loaded in DetectionService
            # We reuse _ensure_engine_system but with a custom check script? 
            # Easier: Just run a small task on engine to check
            
            # Define check function
            async def _check_yolo():
                from src.core.locator import get_active_locator
                from src.ucorefs.detection import DetectionService
                
                # Get Engine Locator (ContextVar should be correct in Engine Thread)
                # But to be safe, we rely on _ensure_engine_system logic or similar
                # Here we are inside the closure sent to proxy.submit()
                
                # Try to get locator from current thread
                from PySide6.QtCore import QThread
                thread = QThread.currentThread()
                sl = getattr(thread, 'locator', None)
                if not sl:
                    sl = get_active_locator()
                
                try:
                    service = sl.get_system(DetectionService)
                    return service.get_backend("yolo") is not None
                except Exception:
                    return False

            future = proxy.submit(_check_yolo())
            import asyncio
            return await asyncio.wrap_future(future)
            
        except Exception as e:
            logger.warning(f"YOLO check failed: {e}")
            return False


# ==================== Helper Functions ====================
    
# Helper task for Engine execution
async def _ensure_engine_system(system_cls_name: str, module_path: str) -> bool:
    """
    Task running on Engine Loop to ensure a system is loaded.
    Imports class dynamically to avoid main-thread import issues in closure.
    """
    import importlib
    from src.core.locator import get_active_locator
    import asyncio
    
    try:
        module = importlib.import_module(module_path)
        system_cls = getattr(module, system_cls_name)
        
        sl = None
        
        # Try to get locator from EngineThread first (bypass contextvar issues)
        try:
            from PySide6.QtCore import QThread
            thread = QThread.currentThread()
            if hasattr(thread, 'locator') and thread.locator:
                sl = thread.locator
        except ImportError:
            pass
            
        if not sl:
            sl = get_active_locator()
            
        instance = sl.get_system(system_cls)
        
        loop = asyncio.get_running_loop()
        
        if hasattr(instance, '_ensure_model_sync'):
            return await loop.run_in_executor(None, instance._ensure_model_sync)
        elif hasattr(instance, '_ensure_model'):
            return await instance._ensure_model()
            
        return True
    except Exception as e:
        logger.error(f"Engine load failed for {system_cls_name}: {e}")
        return False

