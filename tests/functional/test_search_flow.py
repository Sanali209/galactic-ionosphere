import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.core.ai.search import SearchService
from src.core.database.models.image import ImageRecord
from bson import ObjectId

@pytest.fixture
def mock_search_deps():
    v_driver = MagicMock()
    e_service = MagicMock()
    
    e_service.encode_text = AsyncMock(return_value=[0.1]*512)
    v_driver.search = AsyncMock()
    
    return v_driver, e_service

@pytest.mark.asyncio
async def test_search_by_text(db_teardown, mock_search_deps):
    v_driver, e_service = mock_search_deps
    
    # Setup DB Data
    img1 = ImageRecord(path="/tmp", filename="cat.jpg")
    await img1.save()
    img2 = ImageRecord(path="/tmp", filename="dog.jpg")
    await img2.save()
    
    # Mock Qdrant Results
    # Return 2 hits, pointing to img1 and img2
    # Simulate ScoredPoint object
    class MockPoint:
        def __init__(self, payload):
            self.payload = payload
            
    hit1 = MockPoint({"mongo_id": str(img1.id)})
    hit2 = MockPoint({"mongo_id": str(img2.id)})
    
    v_driver.search.return_value = [hit1, hit2]
    
    # Service
    service = SearchService(v_driver, e_service)
    
    # Execute
    results = await service.search_by_text("funny cats")
    
    assert len(results) == 2
    assert results[0].id == img1.id
    assert results[1].id == img2.id
    
    e_service.encode_text.assert_called_with("funny cats")
    v_driver.search.assert_called()
