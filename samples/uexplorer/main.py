"""
UExplorer - Main Entry Point

Directory Opus-inspired file manager showcasing Foundation + UCoreFS.

Refactored to use Foundation's run_app helper and SystemBundles for cleaner startup.
Now with loading dialog showing all startup stages.
"""
import sys
import asyncio
from pathlib import Path

# CRITICAL: Load .env FIRST before any other imports
# This ensures HF_TOKEN is available when extractors are imported
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
        import os
        if os.environ.get("HF_TOKEN"):
            print(f"✓ HF_TOKEN loaded from {_env_path}")
    except ImportError:
        pass

# Add foundation to path (allows running main.py directly for debugging)
foundation_path = Path(__file__).parent.parent.parent
if str(foundation_path) not in sys.path:
    sys.path.insert(0, str(foundation_path))

# Add uexplorer to path
uexplorer_path = Path(__file__).parent
if str(uexplorer_path) not in sys.path:
    sys.path.insert(0, str(uexplorer_path))

from loguru import logger
from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

# Import Foundation bootstrap
from src.core.bootstrap import ApplicationBuilder

# Import system bundles
from src.ucorefs.bundle import UCoreFSBundle
from uexplorer_src.bundle import UExplorerUIBundle

# Import UExplorer UI
from uexplorer_src.ui.main_window import MainWindow
from uexplorer_src.viewmodels.main_viewmodel import MainViewModel
from uexplorer_src.ui.dialogs.loading_dialog import LoadingDialog
from uexplorer_src.startup import StartupOrchestrator


def main():
    """Main entry point with loading dialog."""
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Starting")
    logger.info("=" * 60)
    
    config_path = Path(__file__).parent / "config.json"
    
    # Setup Qt application
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Show loading dialog immediately
    loading_dialog = LoadingDialog()
    loading_dialog.show()
    app.processEvents()
    
    # Keep references to prevent GC
    _keep_alive = {"window": None, "locator": None}

    def ui_setup(service_locator):
        """Phase 2: UI Setup (runs as sync callback)."""
        # Stage 7: UI Setup
        loading_dialog.set_stage("ui", "loading")
        
        from src.ui.mvvm.provider import ViewModelProvider
        provider = ViewModelProvider(service_locator)
        main_vm = provider.get(MainViewModel)
        
        window = MainWindow(main_vm)
        _keep_alive["window"] = window
        
        loading_dialog.set_stage("ui", "done")
        
        # Stage 8: Task System
        loading_dialog.set_stage("tasks", "loading")
        
        try:
            from src.core.tasks.system import TaskSystem
            task_system = service_locator.get_system(TaskSystem)
            
            # Register UExplorer task handlers
            from uexplorer_src.tasks.handlers import register_handlers
            register_handlers(task_system)
            
            # Explicitly schedule workers to start (detached)
            # Since we are in sync callback, we need to get loop carefully
            # But call_soon runs in loop context, so get_running_loop works
            loop = asyncio.get_running_loop()
            loop.create_task(task_system.start_workers())
        except (KeyError, ImportError) as e:
            logger.debug(f"TaskSystem setup skipped: {e}")
        
        loading_dialog.set_stage("tasks", "done")
        
        # Close loading dialog, show main window
        loading_dialog.close()
        window.show()
        
        logger.info("UExplorer started successfully")

    async def async_startup():
        """Phase 1: Preload (Async startup with staged loading)."""
        try:
            from src.ucorefs.bundle import UCoreFSClientBundle
            from src.core.engine.proxy import EngineProxy
            from src.ucorefs.engine_bootstrap import bootstrap_engine
            
            # Stage 1: Database (part of builder.build)
            loading_dialog.set_stage("database", "loading")
            
            # Build application with Client Bundle (UI Only)
            builder = (
                ApplicationBuilder("UExplorer", str(config_path))
                .with_default_systems()
                .with_logging(True)
                .add_bundle(UExplorerUIBundle())
                .add_bundle(UCoreFSClientBundle())  # Read-Only services
            )
            
            loading_dialog.set_stage("database", "done")
            
            # Stage 2: Core services
            loading_dialog.set_stage("core_services", "loading")
            loading_dialog.set_stage("core_services", "done")
            
            # Stage 3: UCoreFS services
            loading_dialog.set_stage("ucorefs_services", "loading")
            
            # Actually build (initializes all services)
            service_locator = await builder.build()
            
            # Keep locator alive
            _keep_alive["locator"] = service_locator
            
            # --- Start Background Engine ---
            # Registers the EngineProxy and boots the Engine Thread
            logger.info("Starting Background Engine...")
            engine_proxy = service_locator.register_system(EngineProxy)
            engine_proxy.set_bootstrap(bootstrap_engine)
            await engine_proxy.initialize()  # Initialize before starting
            await engine_proxy.start_engine()
            # -------------------------------
            
            loading_dialog.set_stage("ucorefs_services", "done")
            
            # Stages 4-6: AI Models
            # StartupOrchestrator will now delegate to EngineProxy
            orchestrator = StartupOrchestrator(loading_dialog)
            await orchestrator.run_preload(service_locator)
            
            # Start Engine Processing (workers) only after models are ready
            logger.info("Starting Engine Task Processing...")
            worker_count = await engine_proxy.start_processing()
            logger.info(f"Engine Task Processing started with {worker_count} workers")
            
            # Schedule Phase 2 (UI Setup) as a CALLBACK
            # This escapes the Task 'async_startup' context completely
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(lambda: ui_setup(service_locator))

        except Exception as e:
            import traceback
            logger.critical(f"Async startup failed: {e}\n{traceback.format_exc()}")
            # Show error dialog
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Startup Error")
            msg.setInformativeText(f"Failed to start application:\n{e}")
            msg.exec_()
            sys.exit(1)
    
    try:
        app.setQuitOnLastWindowClosed(True)
        app.aboutToQuit.connect(loop.stop)
        
        with loop:
            # Schedule startup task
            loop.create_task(async_startup())
            
            # Run event loop (will stop when app.quit() is called)
            loop.run_forever()
            
            # Graceful shutdown
            if _keep_alive["locator"]:
                logger.info("Shutting down services...")
                
                # 1. Shutdown Engine first (stops TaskSystem workers)
                try:
                    from src.core.engine.proxy import EngineProxy
                    engine_proxy = _keep_alive["locator"].get_system(EngineProxy)
                    if engine_proxy:
                        logger.info("Shutting down Engine...")
                        loop.run_until_complete(engine_proxy.shutdown())
                except Exception as e:
                    logger.error(f"Engine shutdown error: {e}")
                
                # 2. Stop all main thread services
                loop.run_until_complete(_keep_alive["locator"].stop_all())
                
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except RuntimeError as e:
        if "Event loop stopped" not in str(e):
            raise
    
    logger.info("👋 UExplorer shutdown complete")


if __name__ == "__main__":
    main()
