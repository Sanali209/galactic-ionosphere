"""
Engine Bootstrap Logic.

Defines the initialization sequence for the background Engine thread.
"""
from typing import Any
from loguru import logger
import asyncio
from src.core.tasks.system import TaskSystem  # Import for signal connections

async def bootstrap_engine(thread) -> Any:
    """
    Bootstrap the Engine ServiceLocator.
    
    This runs inside the Engine Thread.
    
    Args:
        thread: EngineThread instance (provides access to config and signals)
    """
    from src.core.bootstrap import ApplicationBuilder
    from src.core.locator import set_active_locator
    from src.ucorefs.bundle import UCoreFSEngineBundle
    from src.core.database.manager import DatabaseManager
    
    config_path = thread._config_path
    
    logger.info("Bootstrapping Engine with UCoreFSEngineBundle...")
    
    # Build the Engine's ServiceLocator
    # Build the Engine's ServiceLocator
    
    # CRITICAL: Switch to a fresh ServiceLocator context for the Engine Thread
    # Otherwise ApplicationBuilder will mutate the Default Locator shared with Main Thread
    from src.core.locator import ServiceLocator, set_active_locator
    engine_locator = ServiceLocator()
    set_active_locator(engine_locator)
    
    # Build Engine with new bundle architecture
    # Use for_engine preset which includes default systems + logging
    builder = (ApplicationBuilder.for_engine("UExplorerEngine", config_path)
        .add_bundle(UCoreFSEngineBundle()))  # AI, Processing, Maintenance
    
    logger.info("=" * 60)
    logger.info("Starting Engine initialization (TaskSystem will restore pending tasks)...")
    logger.info("=" * 60)
    
    # Build it
    await builder.build()
    
    logger.info("=" * 60)
    logger.info("Engine systems initialized - check above for task recovery logs")
    logger.info("=" * 60)
    
    # CRITICAL: Start Engine TaskSystem workers
    # UPDATE: We now start them manually from Main Thread (via EngineProxy.start_processing)
    # AFTER models are preloaded. This prevents tasks from failing due to missing models.
    # task_system = engine_locator.get_system(TaskSystem)
    # workers_count = await task_system.start_workers()
    # logger.info(f"Engine TaskSystem workers started: {workers_count} workers")
    
    logger.info("Engine Bootstrap: Systems initialized (workers pending explicit start)")
    
    # NOTE: Signal connections are already handled in EngineProxy.initialize()
    # for the Thread -> Proxy link. BUT we must connect TaskSystem -> Thread here.
    try:
        task_system = engine_locator.get_system(TaskSystem)
        # Connect TaskSystem signals to EngineThread signals (Relay)
        task_system.signals.task_started.connect(thread.task_started.emit)
        task_system.signals.task_completed.connect(thread.task_completed.emit)
        task_system.signals.task_failed.connect(thread.task_failed.emit)
        task_system.signals.task_progress.connect(thread.task_progress.emit)
        logger.info("✓ Connected TaskSystem signals to EngineThread")
    except Exception as e:
        logger.error(f"Failed to connect TaskSystem signals: {e}")
    
    logger.info("Engine Bootstrap Complete.")
    return engine_locator
