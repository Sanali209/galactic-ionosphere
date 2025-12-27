"""
Tests for UExplorer Migration

Validates FilePaneDocument and drag & drop integration.
"""
import pytest
from unittest.mock import Mock
from src.ui.documents.file_pane_document import FilePaneDocument
from src.ui.widgets.file_pane import FilePaneWidget


class TestFilePaneDocument:
    """Test file pane as draggable document."""
    
    def setup_method(self):
        """Create file pane document with mock locator."""
        self.mock_locator = Mock()
        self.doc = FilePaneDocument("Browser 1", self.mock_locator)
    
    def test_document_creation(self):
        """Test file pane document initializes."""
        assert self.doc.title == "Browser 1"
        assert self.doc.locator == self.mock_locator
    
    def test_content_creation(self):
        """Test file pane widget created on demand."""
        content = self.doc.create_content()
        assert isinstance(content, FilePaneWidget)
        assert self.doc._pane is not None
    
    def test_pane_property(self):
        """Test access to underlying file pane."""
        self.doc.create_content()
        assert self.doc.pane == self.doc._pane
    
    def test_can_close(self):
        """Test file pane can always be closed."""
        assert self.doc.can_close() is True
    
    def test_document_has_id(self):
        """Verify document has unique ID."""
        doc1 = FilePaneDocument("Browser 1", self.mock_locator)
        doc2 = FilePaneDocument("Browser 2", self.mock_locator)
        
        assert hasattr(doc1, 'id')
        assert hasattr(doc2, 'id')
        assert doc1.id != doc2.id
    
    def test_title_preserved(self):
        """Verify title preserved through operations."""
        doc = FilePaneDocument("Test Browser", self.mock_locator)
        assert doc.title == "Test Browser"
        
        # Title should persist after content creation
        doc.create_content()
        assert doc.title == "Test Browser"


class TestFilePaneIntegration:
    """Test integration with docking system."""
    
    def test_file_pane_wraps_correctly(self):
        """Test FilePaneWidget is properly wrapped."""
        mock_locator = Mock()
        doc = FilePaneDocument("Browser", mock_locator)
        
        # Before content creation
        assert doc.pane is None
        
        # After content creation
        widget = doc.create_content()
        assert doc.pane is not None
        assert doc.pane == widget
    
    def test_multiple_file_panes(self):
        """Test multiple file pane documents can exist."""
        mock_locator = Mock()
        
        doc1 = FilePaneDocument("Left", mock_locator)
        doc2 = FilePaneDocument("Right", mock_locator)
        doc3 = FilePaneDocument("Bottom", mock_locator)
        
        assert doc1.title == "Left"
        assert doc2.title == "Right"
        assert doc3.title == "Bottom"
        
        # All should have unique IDs
        ids = {doc1.id, doc2.id, doc3.id}
        assert len(ids) == 3
