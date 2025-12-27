
import unittest
from unittest.mock import MagicMock
import sys

# --- Mocking Infrastructure ---

# 1. robust Mock class
class MockClass(MagicMock):
    __qualname__ = "MockClass"

# 2. Mock Modules BEFORE they are imported
sys.modules["PySide6"] = MagicMock()
sys.modules["PySide6.QtCore"] = MagicMock()
sys.modules["PySide6.QtWidgets"] = MagicMock()
sys.modules["PySide6.QtGui"] = MagicMock()
sys.modules["PySide6QtAds"] = MagicMock()
sys.modules["qasync"] = MagicMock()
sys.modules["src.core.bootstrap"] = MagicMock()

# 3. Setup QObject/QWidget mocks for inheritance
class MockQObject:
    def __init__(self, *args, **kwargs): pass

qt_widgets = sys.modules["PySide6.QtWidgets"]
qt_core = sys.modules["PySide6.QtCore"]
qt_ads = sys.modules["PySide6QtAds"]

qt_core.QObject = MockQObject
qt_widgets.QWidget = MockQObject
qt_widgets.QMainWindow = MockQObject

# 4. Setup critical return values
qt_ads.CDockManager = MagicMock()
qt_ads.CDockManager.return_value.saveState.return_value = b"MOCK_BYTES"
qt_ads.CDockManager.return_value.restoreState.return_value = True

# --- Import Target ---
try:
    from src.ui.docking.service import DockingService
except ImportError as e:
    print(f"Failed to import DockingService: {e}")
    sys.exit(1)

# --- Test Case ---

class TestDockingIntegration(unittest.TestCase):
    def setUp(self):
        # Create service with a mocked window
        self.mock_window = MagicMock()
        self.service = DockingService(self.mock_window)
        
        # Reset the mock manager for clean assertions
        self.service.dock_manager.reset_mock()
        
        # Ensure helper method behavior matches expected
        self.service.dock_manager.saveState.return_value = b"MOCK_BYTES"

    def test_add_panel_auto_hide(self):
        """Verify adding a panel with auto_hide=True uses addAutoHideDockWidget."""
        print("Testing add_panel(auto_hide=True)...")
        mock_widget = MagicMock()
        self.service.add_panel("test_panel", mock_widget, "Test Panel", area="right", auto_hide=True)
        self.service.dock_manager.addAutoHideDockWidget.assert_called_once()
        print("PASS")
    
    def test_add_panel_normal(self):
        """Verify adding a panel with auto_hide=False uses addDockWidget."""
        print("Testing add_panel(auto_hide=False)...")
        mock_widget = MagicMock()
        self.service.add_panel("test_panel_2", mock_widget, "Panel 2", area="left", auto_hide=False)
        self.service.dock_manager.addDockWidget.assert_called_once()
        self.service.dock_manager.addAutoHideDockWidget.assert_not_called()
        print("PASS")

    def test_save_restore_layout(self):
        """Verify save/restore layout calls underlying QtAds methods."""
        print("Testing save/restore layout...")
        
        # Save
        state = self.service.save_layout()
        self.service.dock_manager.saveState.assert_called_once()
        self.assertEqual(state, b"MOCK_BYTES")
        
        # Restore
        self.service.restore_layout(b"RESTORE_BYTES")
        self.service.dock_manager.restoreState.assert_called_once_with(b"RESTORE_BYTES")
        print("PASS")

if __name__ == '__main__':
    unittest.main()
