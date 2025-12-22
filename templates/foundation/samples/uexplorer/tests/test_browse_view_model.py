"""
Unit Tests for BrowseViewModel

Tests the per-document ViewModel state management.
"""
import pytest
from unittest.mock import MagicMock
from bson import ObjectId


class TestBrowseViewModel:
    """Tests for BrowseViewModel."""
    
    @pytest.fixture
    def viewmodel(self):
        """Create a BrowseViewModel instance."""
        from uexplorer_src.viewmodels.browse_view_model import BrowseViewModel
        return BrowseViewModel(doc_id="test_doc")
    
    def test_init(self, viewmodel):
        """Test initialization."""
        assert viewmodel.doc_id == "test_doc"
        assert viewmodel.results == []
        assert viewmodel.is_loading == False
        assert viewmodel.view_mode == "tree"
        assert viewmodel.selection == []
    
    def test_set_results(self, viewmodel):
        """Test setting results emits signal."""
        received = []
        viewmodel.results_changed.connect(lambda r: received.append(r))
        
        mock_files = [MagicMock(), MagicMock()]
        viewmodel.set_results(mock_files)
        
        assert viewmodel.results == mock_files
        assert len(received) == 1
        assert received[0] == mock_files
    
    def test_set_loading(self, viewmodel):
        """Test loading state changes."""
        received = []
        viewmodel.loading_changed.connect(lambda l: received.append(l))
        
        viewmodel.set_loading(True)
        assert viewmodel.is_loading == True
        assert received == [True]
        
        viewmodel.set_loading(False)
        assert viewmodel.is_loading == False
        assert received == [True, False]
    
    def test_set_view_mode(self, viewmodel):
        """Test view mode changes."""
        received = []
        viewmodel.view_mode_changed.connect(lambda m: received.append(m))
        
        viewmodel.set_view_mode("card")
        assert viewmodel.view_mode == "card"
        assert received == ["card"]
        
        # Invalid mode should not change
        viewmodel.set_view_mode("invalid")
        assert viewmodel.view_mode == "card"
    
    def test_set_selection(self, viewmodel):
        """Test selection changes."""
        received = []
        viewmodel.selection_changed.connect(lambda s: received.append(s))
        
        ids = [ObjectId(), ObjectId()]
        viewmodel.set_selection(ids)
        
        assert viewmodel.selection == ids
        assert len(received) == 1
    
    def test_set_directory(self, viewmodel):
        """Test directory changes."""
        received = []
        viewmodel.directory_changed.connect(lambda d: received.append(d))
        
        dir_id = ObjectId()
        viewmodel.set_directory(dir_id)
        
        assert viewmodel.current_directory == dir_id
        assert len(received) == 1
    
    def test_set_search(self, viewmodel):
        """Test search parameter setting."""
        viewmodel.set_search("test query", mode="vector", fields=["name", "path"])
        
        state = viewmodel.get_query_state()
        assert state["search_text"] == "test query"
        assert state["search_mode"] == "vector"
        assert state["search_fields"] == ["name", "path"]
    
    def test_clear_search(self, viewmodel):
        """Test clearing search."""
        received = []
        viewmodel.results_changed.connect(lambda r: received.append(r))
        
        viewmodel.set_results([MagicMock()])
        viewmodel.clear_search()
        
        assert viewmodel.results == []
        assert len(received) == 2  # Initial results + clear
