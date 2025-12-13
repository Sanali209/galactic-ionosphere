import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    print("Importing MainWindow...")
    from src.ui.main_window import MainWindow
    print("MainWindow imported successfully.")
    
    from src.ui.widgets.gallery import GalleryWidget
    from src.ui.widgets.sidebar import SidebarWidget
    from src.ui.widgets.properties import PropertiesWidget
    from src.ui.widgets.journal import JournalWidget
    print("Widgets imported successfully.")

except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
