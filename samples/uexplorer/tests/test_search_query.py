"""
Unit Tests for SearchQuery

Tests the SearchQuery dataclass.
"""
import pytest
from bson import ObjectId


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""
    
    @pytest.fixture
    def query(self):
        """Create a SearchQuery instance."""
        from uexplorer_src.viewmodels.search_query import SearchQuery
        return SearchQuery
    
    def test_default_values(self, query):
        """Test default values."""
        q = query()
        assert q.text == ""
        assert q.mode == "text"
        assert q.fields == ["name"]
        assert q.file_id is None
        assert q.limit == 100
    
    def test_is_empty(self, query):
        """Test is_empty check."""
        q = query()
        assert q.is_empty() == True
        
        q = query(text="hello")
        assert q.is_empty() == False
    
    def test_is_text_search(self, query):
        """Test text search detection."""
        q = query(text="hello", mode="text")
        assert q.is_text_search() == True
        assert q.is_vector_search() == False
        assert q.is_image_search() == False
    
    def test_is_vector_search(self, query):
        """Test vector search detection."""
        q = query(text="hello", mode="vector")
        assert q.is_vector_search() == True
        assert q.is_text_search() == False
    
    def test_is_image_search(self, query):
        """Test image search detection."""
        file_id = ObjectId()
        q = query(mode="image", file_id=file_id)
        assert q.is_image_search() == True
        assert q.is_text_search() == False
    
    def test_with_filters(self, query):
        """Test query with filters."""
        q = query(
            text="test",
            filters={"file_type": ["image"], "rating": 3},
            tags=[ObjectId()],
            tag_mode="all"
        )
        
        assert q.filters["file_type"] == ["image"]
        assert q.tag_mode == "all"
