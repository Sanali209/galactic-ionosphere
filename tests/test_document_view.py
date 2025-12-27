"""
Unit tests for DocumentView and DocumentViewModel (Phase 1).
"""
import pytest
from unittest.mock import MagicMock
from src.ui.documents.document_view import DocumentView, DocumentViewModel

def test_document_viewmodel_creation():
    """Test creating DocumentViewModel."""
    locator = MagicMock()
    vm = DocumentViewModel(locator, "/path/to/file.txt")
    
    assert vm.file_path == "/path/to/file.txt"
    assert vm.title == "file.txt"
    assert not vm.is_modified

def test_document_viewmodel_untitled():
    """Test DocumentViewModel without file path."""
    locator = MagicMock()
    vm = DocumentViewModel(locator)
    
    assert vm.file_path is None
    assert vm.title == "Untitled"

def test_document_state_modified_flag():
    """Test document modified state."""
    locator = MagicMock()
    vm = DocumentViewModel(locator, "test.txt")
    
    assert not vm.is_modified
    assert vm.title == "test.txt"
    
    vm.mark_modified()
    assert vm.is_modified
    assert vm.title == "test.txt*"  # Asterisk added
    
    vm.mark_saved()
    assert not vm.is_modified
    assert vm.title == "test.txt"

def test_document_view_signals(qapp):
    """Test DocumentView signals."""
    if qapp is None: pytest.skip("No QApp")
    
    locator = MagicMock()
    vm = DocumentViewModel(locator)
    doc = DocumentView(vm)
    
    # Mock signal handlers
    content_changed = MagicMock()
    save_requested = MagicMock()
    
    doc.content_changed.connect(content_changed)
    doc.save_requested.connect(save_requested)
    
    # Emit signals
    doc.content_changed.emit()
    doc.save_requested.emit()
    
    content_changed.assert_called_once()
    save_requested.assert_called_once()

def test_document_save():
    """Test document save marks as not modified."""
    locator = MagicMock()
    vm = DocumentViewModel(locator, "test.txt")
    doc = DocumentView(vm)
    
    vm.mark_modified()
    assert vm.is_modified
    
    doc.save()
    assert not vm.is_modified

def test_document_can_close():
    """Test default can_close behavior."""
    locator = MagicMock()
    vm = DocumentViewModel(locator)
    doc = DocumentView(vm)
    
    assert doc.can_close()  # Default implementation returns True
