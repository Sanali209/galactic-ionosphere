"""
Unit Tests for DocumentManager

Tests multi-document tracking and active document management.
"""
import pytest
from unittest.mock import MagicMock


class TestDocumentManager:
    """Tests for DocumentManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a DocumentManager instance."""
        from uexplorer_src.viewmodels.document_manager import DocumentManager
        return DocumentManager()
    
    def test_init(self, manager):
        """Test initialization."""
        assert manager.active_id is None
        assert manager.documents == {}
    
    def test_create_first_document(self, manager):
        """Test creating first document sets it as active."""
        received = []
        manager.active_changed.connect(lambda d: received.append(d))
        manager.document_added.connect(lambda id, vm: received.append(("added", id)))
        
        vm = manager.create_document("doc1")
        
        assert "doc1" in manager.documents
        assert manager.active_id == "doc1"
        assert manager.active_viewmodel == vm
        assert ("added", "doc1") in received
    
    def test_create_multiple_documents(self, manager):
        """Test creating multiple documents."""
        vm1 = manager.create_document("doc1")
        vm2 = manager.create_document("doc2")
        
        assert len(manager.documents) == 2
        # First document should still be active
        assert manager.active_id == "doc1"
    
    def test_set_active(self, manager):
        """Test setting active document."""
        received = []
        manager.active_changed.connect(lambda d: received.append(d))
        
        manager.create_document("doc1")
        manager.create_document("doc2")
        
        manager.set_active("doc2")
        
        assert manager.active_id == "doc2"
        assert "doc2" in received
    
    def test_set_active_invalid(self, manager):
        """Test setting active to invalid document does nothing."""
        manager.create_document("doc1")
        manager.set_active("nonexistent")
        
        assert manager.active_id == "doc1"
    
    def test_remove_document(self, manager):
        """Test removing a document."""
        received = []
        manager.document_removed.connect(lambda d: received.append(d))
        
        manager.create_document("doc1")
        manager.create_document("doc2")
        
        manager.remove_document("doc1")
        
        assert "doc1" not in manager.documents
        assert "doc1" in received
        # Active should switch to remaining
        assert manager.active_id == "doc2"
    
    def test_remove_last_document(self, manager):
        """Test removing last document clears active."""
        manager.create_document("doc1")
        manager.remove_document("doc1")
        
        assert manager.active_id is None
        assert len(manager.documents) == 0
    
    def test_send_results_to_active(self, manager):
        """Test sending results to active document."""
        vm1 = manager.create_document("doc1")
        vm2 = manager.create_document("doc2")
        
        manager.set_active("doc2")
        
        results = [MagicMock(), MagicMock()]
        manager.send_results_to_active(results)
        
        # Only doc2 should have results
        assert vm2.results == results
        assert vm1.results == []
    
    def test_duplicate_document_returns_existing(self, manager):
        """Test creating duplicate returns existing ViewModel."""
        vm1 = manager.create_document("doc1")
        vm2 = manager.create_document("doc1")
        
        assert vm1 is vm2
        assert len(manager.documents) == 1
