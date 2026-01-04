import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.ucorefs.search.service import SearchQuery, SearchService

@pytest.fixture
def service():
    """Create service with mocked locator."""
    locator = MagicMock()
    config = MagicMock()
    service_obj = SearchService(locator, config)
    return service_obj

@pytest.mark.asyncio
async def test_parse_simple_detection(service):
    """Test parsing 'Person:2'."""
    query = SearchQuery(text="vacation Person:2")
    
    # We need to expose the private method for testing or test via public API
    # Assuming we'll implement a helper method or verify via side effect on query object
    # For now, let's treat it as if we're calling the internal parser directly 
    # (which we will define in the service)
    
    with patch.object(service, '_parse_query_text', wraps=service._parse_query_text) as mock_parser:
        # In real implementation SearchService.search calls this.
        # But here we want to unit test the parsing method itself if possible.
        # Since it's not written yet, we will define the expectation.
        
        cleaned_text, filters = service._parse_query_text("vacation Person:2")
        
        assert cleaned_text.strip() == "vacation"
        assert len(filters) == 1
        assert filters[0]['class_name'].lower() == "person"
        assert filters[0]['min_count'] == 2
        assert filters[0]['negate'] is False

@pytest.mark.asyncio
async def test_parse_complex_detection(service):
    """Test parsing 'Person:face:1' and '!Car'."""
    text = "Person:face:1 !Car beach"
    cleaned_text, filters = service._parse_query_text(text)
    
    assert cleaned_text.strip() == "beach"
    assert len(filters) == 2
    
    # Verify Person:face:1
    person = next(f for f in filters if f['class_name'].lower() == 'person')
    assert person['group_name'].lower() == 'face'
    assert person['min_count'] == 1
    assert person['negate'] is False
    
    # Verify !Car
    car = next(f for f in filters if f['class_name'].lower() == 'car')
    assert car['negate'] is True
    assert car['min_count'] == 1  # Default

@pytest.mark.asyncio
async def test_parse_any_group(service):
    """Test 'Person:any:2'."""
    text = "Person:any:2"
    cleaned_text, filters = service._parse_query_text(text)
    
    assert cleaned_text.strip() == ""
    assert filters[0]['group_name'] == 'any'

@pytest.mark.asyncio
async def test_parse_multiple_same_class(service):
    """Test 'Person:face:1 Person:body:1' (Should be AND)."""
    text = "Person:face:1 Person:body:1"
    cleaned_text, filters = service._parse_query_text(text)
    assert len(filters) == 2

# Mocking the actual resolution logic would require patching DetectionInstance.find
# We will leave that for integration testing logic or trusted implementation.
