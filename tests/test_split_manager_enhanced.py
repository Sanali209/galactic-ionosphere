"""
Tests for Enhanced SplitManager

Validates document movement and drag coordinator integration.
"""
import pytest
from unittest.mock import Mock, MagicMock
from PySide6.QtCore import Qt
from src.ui.documents.split_manager import SplitManager, SplitNode
from src.ui.documents.split_container import SplitContainer


class TestDocumentMovement:
    """Test document movement between containers."""
    
    def setup_method(self):
        self.manager = SplitManager()
        
        # Create mock containers
        self.container1 = Mock(spec=SplitContainer)
        self.container2 = Mock(spec=SplitContainer)
        
        self.manager.root.container_widget = self.container1
    
    def test_move_document_between_containers(self):
        """Test moving document from one container to another."""
        # Setup: Container 1 has a document
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        self.container1.remove_document.return_value = mock_doc
        
        # Create second container via split
        from src.ui.documents.split_manager import SplitOrientation
        container2_id = self.manager.split_node(self.manager.root.id, SplitOrientation.HORIZONTAL)
        node2 = self.manager.get_node(container2_id)
        node2.container_widget = self.container2
        
        # Execute move
        result = self.manager.move_document("doc_1", self.manager.root.id, container2_id)
        
        # Verify
        assert result is True
        self.container1.remove_document.assert_called_once_with("doc_1")
        self.container2.add_document.assert_called_once_with(mock_doc)
    
    def test_move_document_invalid_source(self):
        """Test move fails with invalid source container."""
        result = self.manager.move_document("doc_1", "invalid_id", self.manager.root.id)
        assert result is False
    
    def test_move_document_invalid_target(self):
        """Test move fails with invalid target container."""
        result = self.manager.move_document("doc_1", self.manager.root.id, "invalid_id")
        assert result is False
    
    def test_move_nonexistent_document(self):
        """Test move fails when document doesn't exist."""
        self.container1.remove_document.return_value = None
        
        from src.ui.documents.split_manager import SplitOrientation
        container2_id = self.manager.split_node(self.manager.root.id, SplitOrientation.HORIZONTAL)
        result = self.manager.move_document("nonexistent", self.manager.root.id, container2_id)
        
        assert result is False


class TestSplitWithDocument:
    """Test split creation with document movement."""
    
    def test_split_with_document_horizontal(self):
        """Test creating horizontal split and moving document."""
        manager = SplitManager()
        
        # Mock container
        mock_container = Mock(spec=SplitContainer)
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        mock_container.remove_document.return_value = mock_doc
        manager.root.container_widget = mock_container
        
        # Execute
        from src.ui.documents.split_manager import SplitOrientation
        new_id = manager.split_with_document(manager.root.id, SplitOrientation.HORIZONTAL, "doc_1")
        
        # Verify split created and document moved
        assert new_id is not None
        node = manager.get_node(new_id)
        assert node is not None
        assert node.is_container
    
    def test_split_with_document_vertical(self):
        """Test creating vertical split and moving document."""
        manager = SplitManager()
        
        mock_container = Mock(spec=SplitContainer)
        mock_doc = Mock()
        mock_doc.id = "doc_1"
        mock_container.remove_document.return_value = mock_doc
        manager.root.container_widget = mock_container
        
        from src.ui.documents.split_manager import SplitOrientation
        new_id = manager.split_with_document(manager.root.id, SplitOrientation.VERTICAL, "doc_1")
        
        assert new_id is not None


class TestDragCoordinatorIntegration:
    """Test drag coordinator initialization."""
    
    def test_drag_coordinator_created(self):
        """Verify drag coordinator is initialized."""
        manager = SplitManager()
        assert hasattr(manager, 'drag_coordinator')
        assert manager.drag_coordinator is not None
    
    def test_drag_coordinator_references_manager(self):
        """Verify coordinator has reference to manager."""
        manager = SplitManager()
        assert manager.drag_coordinator.split_manager is manager


class TestEnhancedSplitManager:
    """Test enhanced split manager functionality."""
    
    def test_initialization(self):
        """Test manager initializes with drag coordinator."""
        manager = SplitManager()
        
        assert manager.root is not None
        assert manager.active_node_id == manager.root.id
        assert manager.drag_coordinator is not None
    
    def test_move_document_method_exists(self):
        """Verify move_document method exists."""
        manager = SplitManager()
        assert hasattr(manager, 'move_document')
        assert callable(manager.move_document)
    
    def test_split_with_document_method_exists(self):
        """Verify split_with_document method exists."""
        manager = SplitManager()
        assert hasattr(manager, 'split_with_document')
        assert callable(manager.split_with_document)
