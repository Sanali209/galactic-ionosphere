"""
Image Search Application - Entry Point

Simplified version using Foundation bootstrap helpers.
"""
import sys
from pathlib import Path

# Add foundation to path temporarily (until pip install -e . is run)
APP_DIR = Path(__file__).parent
FOUNDATION_DIR = APP_DIR.parent.parent / "templates/foundation"
sys.path.insert(0, str(FOUNDATION_DIR))

# Import from foundation package
from foundation import ApplicationBuilder, run_app

# Import local app modules
from src.core.search_service import SearchService
from src.ui.viewmodels.main_viewmodel import MainViewModel
from src.ui.main_window import MainWindow


if __name__ == "__main__":
    # Configure application
    builder = (ApplicationBuilder("Image Search", str(APP_DIR / "config.json"))
               .with_default_systems()
               .with_logging(True)
               .add_system(SearchService))
    
    # Run application  
    run_app(MainWindow, MainViewModel, builder=builder)

