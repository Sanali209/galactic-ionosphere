"""
Foundation Application - Demo Entry Point

Demonstrates the simplified bootstrap pattern for Foundation apps.
"""
from src.core.bootstrap import ApplicationBuilder, run_app
from src.ui.main_window import MainWindow
from src.ui.viewmodels.main_viewmodel import MainViewModel


if __name__ == "__main__":
    # Configure application with fluent builder API
    builder = (ApplicationBuilder("Foundation App", "config.json")
               .with_default_systems()
               .with_logging(True))
    
    # Run application with one line
    run_app(MainWindow, MainViewModel, builder=builder)


