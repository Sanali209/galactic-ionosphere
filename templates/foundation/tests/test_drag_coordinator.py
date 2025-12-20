"""
Tests for Drag Coordinator

Validates drag operation coordination across containers.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtCore import Qt
from src.ui.documents.docking.drag_coordinator import DragCoordinator
from src.ui.documents.docking.drop_zone_overlay import DropZone
from src.ui.documents.split_manager import SplitManager


class TestDragCoordinator:
    """Test drag operation coordination."""
    
    def setup_method(self):
        """Create coordinator with mock split manager."""
        self.split_manager = SplitManager()
        self.coordinator = DragCoordinator(self.split_manager)
    
    def test_initialization(self):
        """Test coordinator initializes correctly."""
        assert self.coordinator.split_manager == self.split_manager
        assert self.coordinator.active_drag is None
        assert len(self.coordinator.overlays) == 0
        assert len(self.coordinator.previews) == 0
    
    def test_start_drag_creates_overlays(self):
        """Test overlays created for all containers."""
        # SplitManager starts with one container (root)
        container_id = self.split_manager.root.id
        
        # Mock container widget
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        self.split_manager.root.container_widget = mock_widget
        
        # Start drag
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay') as MockOverlay:
            mock_overlay = Mock()
            MockOverlay.return_value = mock_overlay
            
            self.coordinator.start_drag("doc_1", container_id)
            
            # Verify overlay created
            assert MockOverlay.called
            assert mock_overlay.show.called
    
    def test_active_drag_state(self):
        """Test drag state tracking."""
        # Mock container
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        self.split_manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            self.coordinator.start_drag("doc_1", "container_1")
        
        assert self.coordinator.active_drag is not None
        assert self.coordinator.active_drag['document_id'] == "doc_1"
        assert self.coordinator.active_drag['source'] == "container_1"
    
    def test_cleanup_on_end_drag(self):
        """Test overlays cleaned up when drag ends."""
        container_id = self.split_manager.root.id
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        self.split_manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            self.coordinator.start_drag("doc_1", container_id)
        
        # End drag
        self.coordinator.end_drag()
        
        assert len(self.coordinator.overlays) == 0
        assert len(self.coordinator.previews) == 0
        assert self.coordinator.active_drag is None
    
    def test_move_to_center_zone(self):
        """Test drop on CENTER zone moves document."""
        container_id = self.split_manager.root.id
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        self.split_manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            self.coordinator.start_drag("doc_1", container_id)
        
        with patch.object(self.coordinator, '_move_document') as mock_move:
            self.coordinator._execute_drop(container_id, DropZone.CENTER)
            mock_move.assert_called_once_with("doc_1", container_id, container_id)
    
    def test_drop_on_edge_creates_split(self):
        """Test drop on edge zone creates new split."""
        container_id = self.split_manager.root.id
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        self.split_manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            self.coordinator.start_drag("doc_1", container_id)
        
        with patch.object(self.split_manager, 'split_node', return_value="new_id") as mock_split:
            self.coordinator._execute_drop(container_id, DropZone.LEFT)
            mock_split.assert_called_once_with(container_id, Qt.Horizontal)


class TestPreviewCalculation:
    """Test preview rectangle calculations."""
    
    def setup_method(self):
        self.manager = SplitManager()
        self.coordinator = DragCoordinator(self.manager)
        
        # Setup mock widget
        mock_widget = Mock()
        mock_widget.width.return_value = 800
        mock_widget.height.return_value = 600
        self.manager.root.container_widget = mock_widget
    
    def test_center_preview_full_size(self):
        """CENTER preview should be full container size."""
        rect = self.coordinator._calculate_preview_rect(self.manager.root.id, DropZone.CENTER)
        assert rect.width() == 800
        assert rect.height() == 600
    
    def test_left_preview_half_width(self):
        """LEFT preview should be left half."""
        rect = self.coordinator._calculate_preview_rect(self.manager.root.id, DropZone.LEFT)
        assert rect.x() == 0
        assert rect.width() == 400
        assert rect.height() == 600
    
    def test_right_preview_half_width(self):
        """RIGHT preview should be right half."""
        rect = self.coordinator._calculate_preview_rect(self.manager.root.id, DropZone.RIGHT)
        assert rect.x() == 400
        assert rect.width() == 400
        assert rect.height() == 600
    
    def test_top_preview_half_height(self):
        """TOP preview should be top half."""
        rect = self.coordinator._calculate_preview_rect(self.manager.root.id, DropZone.TOP)
        assert rect.y() == 0
        assert rect.width() == 800
        assert rect.height() == 300
    
    def test_bottom_preview_half_height(self):
        """BOTTOM preview should be bottom half."""
        rect = self.coordinator._calculate_preview_rect(self.manager.root.id, DropZone.BOTTOM)
        assert rect.y() == 300
        assert rect.width() == 800
        assert rect.height() == 300


class TestDropExecution:
    """Test drop operation execution."""
    
    def test_horizontal_split_on_left_drop(self):
        """LEFT drop creates horizontal split."""
        manager = SplitManager()
        coordinator = DragCoordinator(manager)
        
        container_id = manager.root.id
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            coordinator.start_drag("doc_1", container_id)
        
        with patch.object(manager, 'split_node', return_value="new_id") as mock_split:
            coordinator._execute_drop(container_id, DropZone.LEFT)
            # Verify Qt.Horizontal used for LEFT/RIGHT
            assert mock_split.call_args[0][1] == Qt.Horizontal
    
    def test_vertical_split_on_top_drop(self):
        """TOP drop creates vertical split."""
        manager = SplitManager()
        coordinator = DragCoordinator(manager)
        
        container_id = manager.root.id
        mock_widget = Mock()
        mock_widget.size.return_value = Mock(width=lambda: 800, height=lambda: 600)
        manager.root.container_widget = mock_widget
        
        with patch('src.ui.documents.docking.drag_coordinator.DropZoneOverlay'):
            coordinator.start_drag("doc_1", container_id)
        
        with patch.object(manager, 'split_node', return_value="new_id") as mock_split:
            coordinator._execute_drop(container_id, DropZone.TOP)
            # Verify Qt.Vertical used for TOP/BOTTOM
            assert mock_split.call_args[0][1] == Qt.Vertical
