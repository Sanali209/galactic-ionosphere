import pytest
from unittest.mock import MagicMock, patch
from src.core.ai.vector_driver import VectorDriver

@pytest.fixture
def mock_qdrant():
    with patch("src.core.ai.vector_driver.QdrantClient") as MockClient:
        yield MockClient

def test_driver_connect(mock_qdrant):
    # Setup
    driver = VectorDriver("localhost", 6333)
    
    # Action
    driver.connect()
    
    # Assert
    mock_qdrant.assert_called_with(host="localhost", port=6333)
    instance = mock_qdrant.return_value
    instance.collection_exists.assert_called()

def test_driver_ensure_collection_creates_if_missing(mock_qdrant):
    driver = VectorDriver("localhost", 6333)
    driver.connect()
    
    instance = mock_qdrant.return_value
    # Setup: collection_exists returns False
    instance.collection_exists.return_value = False
    
    driver._ensure_collection()
    
    instance.create_collection.assert_called_once()
    args, kwargs = instance.create_collection.call_args
    assert kwargs['collection_name'] == "gallery"
    assert kwargs['vectors_config'].size == 512
