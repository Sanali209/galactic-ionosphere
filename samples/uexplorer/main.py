"""
UExplorer - Main Entry Point

Directory Opus-inspired file manager showcasing Foundation + UCoreFS.

Refactored to use Foundation's run_app() helper and SystemBundles for cleaner startup.
"""
import sys
from pathlib import Path

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
from src.ucorefs.bundle import UCoreFSBundle
from uexplorer_src.bundle import UExplorerUIBundle

# Import UExplorer UI
from uexplorer_src.ui.main_window import MainWindow
from uexplorer_src.viewmodels.main_viewmodel import MainViewModel


def main():
    """Main entry point using Foundation's run_app helper."""
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Starting")
    logger.info("=" * 60)
    
    config_path = Path(__file__).parent / "config.json"
    
    # Build application with system bundles
    # Bundles encapsulate system registration in correct dependency order
    builder = (
        ApplicationBuilder("UExplorer", str(config_path))
        .with_default_systems()
        .with_logging(True)
        .add_bundle(UExplorerUIBundle())  # SessionState, NavigationService
        .add_bundle(UCoreFSBundle())      # All UCoreFS services (16 systems)
    )
    
    # Run with Foundation's helper (handles Qt, async, shutdown)
    run_app(MainWindow, MainViewModel, builder=builder)
    
    logger.info("👋 UExplorer shutdown complete")


if __name__ == "__main__":
    main()
