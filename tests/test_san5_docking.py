
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Create robust mocks
class MockClass(MagicMock):
    __qualname__ = "MockClass"

# Mock PySide6 and dependencies logic BEFORE imports
sys.modules["PySide6"] = MagicMock()
sys.modules["PySide6.QtCore"] = MagicMock()
sys.modules["PySide6.QtWidgets"] = MagicMock()
sys.modules["PySide6.QtGui"] = MagicMock()
sys.modules["PySide6QtAds"] = MagicMock()
sys.modules["qasync"] = MagicMock() # Block qasync import entirely

# Setup basic Qt logic
QtWidgets = sys.modules["PySide6.QtWidgets"]
QtCore = sys.modules["PySide6.QtCore"]
QtAds = sys.modules["PySide6QtAds"]

# Mock QObject for multiple inheritance
class MockQObject:
    def __init__(self, *args, **kwargs): pass

QtCore.QObject = MockQObject
QtWidgets.QWidget = MockQObject
QtWidgets.QMainWindow = MockQObject
QtAds.CDockManager = MagicMock()
QtAds.CDockManager.saveState.return_value = b"MOCK_STATE"

# Prevent src.core.bootstrap from failing if it's imported
sys.modules["src.core.bootstrap"] = MagicMock()

# Now import the target service
# We need to bypass src/__init__.py if it does too much
from src.ui.docking.service import DockingService

class TestDockingIntegration(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        QtAds.CDockManager.reset_mock()
        self.mock_window = MagicMock()
        self.service = DockingService(self.mock_window)
        
        # Setup dock manager mock behavior
        self.service.dock_manager.saveState.return_value = b"MOCK_STATE_BYTES"
        self.service.dock_manager.restoreState.return_value = True

    def test_add_panel_auto_hide(self):
        """Verify adding a panel with auto_hide=True uses addAutoHideDockWidget."""
        mock_widget = MagicMock()
        
        self.service.add_panel("test_panel", mock_widget, "Test Panel", area="right", auto_hide=True)
        
        # Verify call to addAutoHideDockWidget
        self.service.dock_manager.addAutoHideDockWidget.assert_called_once()
    
    def test_add_panel_normal(self):
        """Verify adding a panel with auto_hide=False uses addDockWidget."""
        mock_widget = MagicMock()
        
        self.service.add_panel("test_panel_2", mock_widget, "Panel 2", area="left", auto_hide=False)
        
        self.service.dock_manager.addDockWidget.assert_called_once()
        self.service.dock_manager.addAutoHideDockWidget.assert_not_called()

    def test_save_restore_layout(self):
        """Verify save/restore layout calls underlying QtAds methods."""
        
        # Test Save
        state = self.service.save_layout()
        self.service.dock_manager.saveState.assert_called_once()
        self.assertEqual(state, b"MOCK_STATE_BYTES")
        
        # Test Restore
        self.service.restore_layout(b"RESTORE_BYTES")
        self.service.dock_manager.restoreState.assert_called_once_with(b"RESTORE_BYTES")

if __name__ == '__main__':
    unittest.main()
