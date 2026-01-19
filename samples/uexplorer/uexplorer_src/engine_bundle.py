"""
UExplorer Engine Integration Bundle.

Integrates the background processing engine with UExplorer main application.
Handles engine startup, model preloading, and task system initialization.
"""
from typing import TYPE_CHECKING
from src.core.bootstrap import SystemBundle

if TYPE_CHECKING:
    from src.core.bootstrap import ApplicationBuilder
    from src.core.service_locator import ServiceLocator


class EngineIntegrationBundle(SystemBundle):
    """
    Integrates background engine with UExplorer.
    
    Registers:
    - EngineProxy: Bridge to background thread
    
    Use with post_build hook to start engine after services initialized.
    
    Example:
        builder = (ApplicationBuilder.for_gui("UExplorer")
            .add_bundle(EngineIntegrationBundle()))
        
        run_app(
            MainWindow,
            MainViewModel,
            builder=builder,
            post_build=start_engine
        )
    """
    
    def register(self, builder: "ApplicationBuilder") -> None:
        """Register EngineProxy service."""
        from src.core.engine.proxy import EngineProxy
        builder.add_system(EngineProxy)


async def start_engine(locator: "ServiceLocator") -> None:
    """
    Post-build hook: Start engine and initialize AI models.
    
    This function is called by run_app() after builder.build() completes.
    
    Steps:
    1. Configure and start engine thread
    2. Wait for engine to become ready
    3. Preload AI models (CLIP, BLIP, GroundingDINO, etc.)
    4. Start task processing workers
    5. Register UI task handlers
    
    Args:
        locator: Main thread ServiceLocator
    """
    from src.core.engine.proxy import EngineProxy
    from src.ucorefs.engine_bootstrap import bootstrap_engine
    from uexplorer_src.startup import StartupOrchestrator
    from uexplorer_src.tasks.handlers import register_handlers
    from src.core.tasks.system import TaskSystem
    from loguru import logger
    
    logger.info("=" * 60)
    logger.info("Starting Background Engine")
    logger.info("=" * 60)
    
    # 1. Configure and start engine thread
    logger.info("Configuring engine...")
    engine_proxy = locator.get_system(EngineProxy)
    engine_proxy.set_bootstrap(bootstrap_engine)
    await engine_proxy.initialize()
    await engine_proxy.start_engine()
    
    # 2. Wait for engine to become ready
    logger.info("Waiting for engine ready...")
    ready = await engine_proxy.wait_for_ready(timeout=30.0)
    if not ready:
        logger.error("Engine failed to become ready within 30 seconds!")
        logger.error("Background processing will not be available.")
        return
    
    logger.info("✓ Engine is ready")
    
    # 3. Preload AI models (silently, no loading dialog)
    logger.info("Preloading AI models...")
    orchestrator = StartupOrchestrator(dialog=None)  # No UI dialog
    success = await orchestrator.run_preload(locator)
    
    if success:
        logger.info("✓ AI models preloaded")
    else:
        logger.warning("Some AI models failed to preload (check logs)")
    
    # 4. Start task processing workers
    logger.info("Starting task processing workers...")
    worker_count = await engine_proxy.start_processing()
    logger.info(f"✓ {worker_count} workers started")
    
    # 5. Register UI task handlers
    logger.info("Registering UI task handlers...")
    task_system = locator.get_system(TaskSystem)
    handler_count = register_handlers(task_system)
    logger.info(f"✓ {handler_count} handlers registered")
    
    logger.info("=" * 60)
    logger.info("✓ Engine Initialization Complete")
    logger.info("=" * 60)
