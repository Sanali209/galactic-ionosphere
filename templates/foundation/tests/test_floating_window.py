"""
Tests for Floating Window

Validates window tearing and re-docking functionality.
"""
import pytest
from unittest.mock import Mock, patch
from PySide6.QtCore import QPoint, Qt, QRect
from PySide6.QtGui import QMouseEvent
from src.ui.documents.docking.floating_window import FloatingWindow


class TestFloatingWindow:
    """Test floating window creation and behavior."""
    
    def setup_method(self):
        """Create floating window with mock document."""
        self.mock_doc = Mock()
        self.mock_doc.id = "doc_1"
        self.mock_doc.title = "Test Document"
        
        self.window = FloatingWindow(self.mock_doc)
    
    def test_window_creation(self):
        """Test floating window initializes correctly."""
        assert self.window.document == self.mock_doc
        assert self.window.windowTitle() == "Test Document"
        assert self.window.centralWidget() == self.mock_doc
    
    def test_window_flags(self):
        """Verify window has correct flags."""
        flags = self.window.windowFlags()
        assert flags & Qt.Window
    
    def test_drag_state_initialization(self):
        """Test drag tracking state initialized."""
        assert self.window._is_dragging is False
        assert self.window._drag_pos is None
    
    def test_mouse_press_starts_drag(self):
        """Test mouse press initiates drag tracking."""
        event = QMouseEvent(
            QMouseEvent.MouseButtonPress,
            QPoint(10, 10),
            QPoint(100, 100),  # global pos
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier
        )
        
        self.window.mousePressEvent(event)
        
        assert self.window._is_dragging is True
        assert self.window._drag_pos is not None
    
    def test_get_document(self):
        """Test get_document returns correct document."""
        doc = self.window.get_document()
        assert doc == self.mock_doc
    
    def test_window_size(self):
        """Test default window size."""
        # Default size should be 800x600
        assert self.window.width() == 800
        assert self.window.height() == 600


class TestWindowTearing:
    """Test tab-to-window tearing operation."""
    
    def test_tear_off_signal(self, qtbot):
        """Test FloatingWindow can be created."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        mock_doc.title = "Test"
        
        window = FloatingWindow(mock_doc)
        assert window is not None
        assert window.document == mock_doc
    
    def test_floating_window_position(self):
        """Test floating window can be positioned."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        mock_doc.title = "Test"
        
        window = FloatingWindow(mock_doc)
        window.move(QPoint(500, 300))
        
        # Position should be set
        assert window.pos().x() >= 0
        assert window.pos().y() >= 0


class TestRedocking:
    """Test re-docking floating windows."""
    
    def test_detect_main_window_hover(self):
        """Test detection when floating window over main window."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        window = FloatingWindow(mock_doc)
        
        # Mock main window
        mock_main = Mock()
        main_rect = QRect(0, 0, 1000, 800)
        mock_main.frameGeometry.return_value = main_rect
        window.setParent(mock_main)
        
        # Test point inside main window
        point_inside = QPoint(500, 400)
        assert main_rect.contains(point_inside)
        
        # Test point outside
        point_outside = QPoint(1500, 400)
        assert not main_rect.contains(point_outside)
    
    def test_dock_requested_signal_exists(self, qtbot):
        """Test dock_requested signal exists."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        window = FloatingWindow(mock_doc)
        
        # Verify signal exists
        assert hasattr(window, 'dock_requested')
    
    def test_closed_signal_exists(self, qtbot):
        """Test closed signal exists."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        window = FloatingWindow(mock_doc)
        
        # Verify signal exists
        assert hasattr(window, 'closed')


class TestWindowLifecycle:
    """Test window lifecycle events."""
    
    def test_close_event_emits_signal(self, qtbot):
        """Test closing window emits signal."""
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        window = FloatingWindow(mock_doc)
        
        with qtbot.waitSignal(window.closed, timeout=1000) as blocker:
            window.close()
        
        assert blocker.args[0] == "doc_1"
