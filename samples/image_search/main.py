"""
Image Search Application - Entry Point

Uses Foundation bootstrap with CardView gallery.
"""
import sys
from pathlib import Path

# Add Foundation to path (until pip install -e . is available)
APP_DIR = Path(__file__).parent
FOUNDATION_DIR = APP_DIR.parent.parent  # templates/foundation
sys.path.insert(0, str(FOUNDATION_DIR))

# Add local app_src to path for app modules
sys.path.insert(0, str(APP_DIR))

# Foundation imports
from src.core.bootstrap import ApplicationBuilder, run_app

# Local imports (from app_src)
from app_src.core.search_service import SearchService
from app_src.ui.viewmodels.main_viewmodel import MainViewModel
from app_src.ui.main_window import MainWindow


if __name__ == "__main__":
    # Configure application
    builder = (
        ApplicationBuilder("Image Search", str(APP_DIR / "config.json"))
        .with_default_systems()
        .with_logging(True)
        .add_system(SearchService)
    )
    
    # Run application  
    run_app(MainWindow, MainViewModel, builder=builder)
