"""
UExplorer - Main Entry Point

Directory Opus-inspired file manager showcasing Foundation + UCoreFS.
Refactored to use Foundation's run_app helper with lifecycle hooks.
"""
import sys
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
            print(f"HF_TOKEN loaded from {_env_path}")
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

# Import Foundation bootstrap
from src.core.bootstrap import ApplicationBuilder, run_app

# Import system bundles
from src.ucorefs.bundles import UCoreFSDataBundle
from src.ui.pyside_bundle import PySideBundle
from uexplorer_src.bundle import UExplorerUIBundle, EngineIntegrationBundle, start_engine

# Import UExplorer UI
from uexplorer_src.ui.main_window import MainWindow
from uexplorer_src.viewmodels.main_viewmodel import MainViewModel


def post_window_init(window, locator):
    """
    Post-window hook: Load initial roots after UI is shown.
    
    Args:
        window: MainWindow instance
        locator: ServiceLocator instance
    """
    window.load_initial_roots()


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Starting")
    logger.info("=" * 60)
    
    config_path = Path(__file__).parent / "config.json"
    
    # Build application with new bundle architecture
    builder = (ApplicationBuilder.for_gui("UExplorer", str(config_path))
        .add_bundle(UCoreFSDataBundle())        # Framework-agnostic data layer
        .add_bundle(PySideBundle())             # Qt/UI framework
        .add_bundle(EngineIntegrationBundle())  # Background engine
        .add_bundle(UExplorerUIBundle()))       # App-specific UI
    
    # Run with lifecycle hooks
    run_app(
        MainWindow,
        MainViewModel,
        builder=builder,
        post_build=start_engine,      # Start engine after services ready
        post_window=post_window_init  # Load roots after window shown
    )


if __name__ == "__main__":
    main()
