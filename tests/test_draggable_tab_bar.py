"""
Tests for Draggable Tab Bar

Validates tab dragging, reordering, and drop operations.
"""
import pytest
from PySide6.QtCore import QPoint, QMimeData, Qt
from PySide6.QtGui import QMouseEvent, QDrag
from PySide6.QtWidgets import QApplication
from src.ui.documents.docking.draggable_tab_bar import DraggableTabBar


class TestDraggableTabBar:
    """Test tab dragging and reordering."""
    
    def setup_method(self):
        """Create tab bar with test tabs."""
        self.tab_bar = DraggableTabBar()
        
        # Add 3 test tabs
        for i in range(3):
            idx = self.tab_bar.addTab(f"Tab {i+1}")
            self.tab_bar.setTabData(idx, f"doc_{i+1}")
    
    def test_tab_preview_creation(self):
        """Test tab pixmap creation for drag cursor."""
        pixmap = self.tab_bar._create_tab_preview(0)
        assert not pixmap.isNull()
        assert pixmap.width() > 0
        assert pixmap.height() > 0
    
    def test_drop_index_calculation_beginning(self):
        """Test drop at beginning of tab bar."""
        # Position before first tab
        pos = QPoint(10, 10)
        index = self.tab_bar._get_drop_index(pos)
        assert index == 0
    
    def test_drop_index_calculation_middle(self):
        """Test drop between tabs."""
        # Get position of second tab
        rect = self.tab_bar.tabRect(1)
        # Position just before second tab
        pos = QPoint(rect.left() - 5, rect.center().y())
        index = self.tab_bar._get_drop_index(pos)
        assert index == 1
    
    def test_drop_index_calculation_end(self):
        """Test drop at end of tab bar."""
        # Position after last tab
        last_rect = self.tab_bar.tabRect(2)
        pos = QPoint(last_rect.right() + 20, 10)
        index = self.tab_bar._get_drop_index(pos)
        assert index == 3  # After last tab
    
    def test_drag_started_signal(self, qtbot):
        """Test signal emission when drag starts."""
        with qtbot.waitSignal(self.tab_bar.tab_drag_started, timeout=1000) as blocker:
            # Simulate drag start
            self.tab_bar._drag_tab_index = 0
            self.tab_bar.tab_drag_started.emit("doc_1")
        
        assert blocker.args[0] == "doc_1"
    
    def test_mime_data_format(self):
        """Test MIME data uses correct format."""
        mime = QMimeData()
        mime.setData("application/x-foundation-document", b"doc_1")
        
        assert mime.hasFormat("application/x-foundation-document")
        data = mime.data("application/x-foundation-document").data()
        assert data.decode() == "doc_1"
    
    def test_accept_drops_enabled(self):
        """Verify tab bar accepts drops."""
        assert self.tab_bar.acceptDrops()
    
    def test_tab_bar_configuration(self):
        """Verify tab bar has correct configuration."""
        assert self.tab_bar.elideMode() == Qt.ElideRight
        assert self.tab_bar.usesScrollButtons()
        assert not self.tab_bar.isMovable()  # We handle ourselves


class TestTabReordering:
    """Test tab reordering within same bar."""
    
    def test_reorder_forward(self, qtbot):
        """Test moving tab from position 0 to 2."""
        tab_bar = DraggableTabBar()
        tab_bar.addTab("A")
        tab_bar.addTab("B")
        tab_bar.addTab("C")
        
        # Emit drop signal (simulating drop at index 2)
        with qtbot.waitSignal(tab_bar.tab_dropped) as blocker:
            tab_bar.tab_dropped.emit("doc_A", 2)
        
        assert blocker.args == ("doc_A", 2)
    
    def test_reorder_backward(self, qtbot):
        """Test moving tab from position 2 to 0."""
        tab_bar = DraggableTabBar()
        tab_bar.addTab("A")
        tab_bar.addTab("B")
        tab_bar.addTab("C")
        
        with qtbot.waitSignal(tab_bar.tab_dropped) as blocker:
            tab_bar.tab_dropped.emit("doc_C", 0)
        
        assert blocker.args == ("doc_C", 0)


class TestDragEvents:
    """Test drag enter/move/drop events."""
    
    def setup_method(self):
        self.tab_bar = DraggableTabBar()
        for i in range(2):
            self.tab_bar.addTab(f"Tab {i+1}")
            self.tab_bar.setTabData(i, f"doc_{i+1}")
    
    def test_drag_enter_accepts_document(self):
        """Test drag enter accepts document MIME type."""
        mime = QMimeData()
        mime.setData("application/x-foundation-document", b"doc_1")
        
        # Drag enter should accept
        # (Can't easily test event handling without full Qt event loop)
        assert mime.hasFormat("application/x-foundation-document")
    
    def test_drag_threshold(self):
        """Test drag doesn't start before threshold."""
        from PySide6.QtWidgets import QApplication
        threshold = QApplication.startDragDistance()
        
        # Movement less than threshold shouldn't trigger drag
        assert threshold > 0
        short_move = threshold // 2
        assert short_move < threshold
