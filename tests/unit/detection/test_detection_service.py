"""
Unit tests for DetectionService group_name handling and bbox validation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from src.ucorefs.detection.service import DetectionService
from src.ucorefs.detection.models import DetectionInstance, DetectionClass


@pytest.fixture
def detection_service():
    """Create a DetectionService instance for testing."""
    locator = MagicMock()
    config = MagicMock()
    service = DetectionService(locator, config)
    return service


@pytest.mark.asyncio
async def test_detect_preserves_group_name_from_backend(detection_service):
    """Test that DetectionService.detect() preserves group_name from backend."""
    # Mock file record
    mock_file = MagicMock()
    mock_file.path = "/test/image.jpg"
    
    # Mock backend that returns group_name
    mock_backend = AsyncMock()
    mock_backend.detect = AsyncMock(return_value=[
        {
            "label": "car",
            "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            "confidence": 0.95,
            "group_name": "sedan"  # Backend provides group_name
        }
    ])
    
    detection_service._backends = {"test_backend": mock_backend}
    
    # Mock database calls
    with patch('src.ucorefs.detection.service.FileRecord') as mock_file_record, \
         patch.object(detection_service, '_get_or_create_class', return_value=MagicMock(_id=ObjectId())):
        
        mock_file_record.get = AsyncMock(return_value=mock_file)
        
        # Execute
        instances = await detection_service.detect(
            file_id=ObjectId(),
            backend="test_backend",
            save=False
        )
        
        # Verify
        assert len(instances) == 1
        assert instances[0].group_name == "sedan"
        assert instances[0].bbox == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


@pytest.mark.asyncio
async def test_detect_defaults_group_name_to_unknown(detection_service):
    """Test that detect() defaults group_name to 'unknown' when not provided."""
    mock_file = MagicMock()
    mock_file.path = "/test/image.jpg"
    
    # Backend doesn't provide group_name
    mock_backend = AsyncMock()
    mock_backend.detect = AsyncMock(return_value=[
        {
            "label": "person",
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
            "confidence": 0.88
            # No group_name field
        }
    ])
    
    detection_service._backends = {"test_backend": mock_backend}
    
    with patch('src.ucorefs.detection.service.FileRecord') as mock_file_record, \
         patch.object(detection_service, '_get_or_create_class', return_value=MagicMock(_id=ObjectId())):
        
        mock_file_record.get = AsyncMock(return_value=mock_file)
        
        instances = await detection_service.detect(
            file_id=ObjectId(),
            backend="test_backend",
            save=False
        )
        
        assert len(instances) == 1
        assert instances[0].group_name == "unknown"


@pytest.mark.asyncio
async def test_bbox_validation_with_empty_dict(detection_service):
    """Test that empty bbox dict is handled gracefully."""
    mock_file = MagicMock()
    mock_file.path = "/test/image.jpg"
    
    mock_backend = AsyncMock()
    mock_backend.detect = AsyncMock(return_value=[
        {
            "label": "dog",
            "bbox": {},  # Empty bbox
            "confidence": 0.75,
            "group_name": "dog"
        }
    ])
    
    detection_service._backends = {"test_backend": mock_backend}
    
    with patch('src.ucorefs.detection.service.FileRecord') as mock_file_record, \
         patch.object(detection_service, '_get_or_create_class', return_value=MagicMock(_id=ObjectId())):
        
        mock_file_record.get = AsyncMock(return_value=mock_file)
        
        instances = await detection_service.detect(
            file_id=ObjectId(),
            backend="test_backend",
            save=False
        )
        
        # Should default to zero bbox
        assert instances[0].bbox == {"x": 0, "y": 0, "w": 0, "h": 0}


@pytest.mark.asyncio
async def test_bbox_validation_with_partial_data(detection_service):
    """Test that partial bbox data is replaced with zeros."""
    mock_file = MagicMock()
    mock_file.path = "/test/image.jpg"
    
    mock_backend = AsyncMock()
    mock_backend.detect = AsyncMock(return_value=[
        {
            "label": "cat",
            "bbox": {"x": 0.1, "y": 0.2},  # Missing w and h
            "confidence": 0.82,
            "group_name": "cat"
        }
    ])
    
    detection_service._backends = {"test_backend": mock_backend}
    
    with patch('src.ucorefs.detection.service.FileRecord') as mock_file_record, \
         patch.object(detection_service, '_get_or_create_class', return_value=MagicMock(_id=ObjectId())):
        
        mock_file_record.get = AsyncMock(return_value=mock_file)
        
        instances = await detection_service.detect(
            file_id=ObjectId(),
            backend="test_backend",
            save=False
        )
        
        # Should replace with zero bbox
        assert instances[0].bbox == {"x": 0, "y": 0, "w": 0, "h": 0}


@pytest.mark.asyncio
async def test_bbox_validation_with_none_value(detection_service):
    """Test that None bbox value is handled."""
    mock_file = MagicMock()
    mock_file.path = "/test/image.jpg"
    
    mock_backend = AsyncMock()
    mock_backend.detect = AsyncMock(return_value=[
        {
            "label": "bird",
            "bbox": None,  # None value
            "confidence": 0.91,
            "group_name": "bird"
        }
    ])
    
    detection_service._backends = {"test_backend": mock_backend}
    
    with patch('src.ucorefs.detection.service.FileRecord') as mock_file_record, \
         patch.object(detection_service, '_get_or_create_class', return_value=MagicMock(_id=ObjectId())):
        
        mock_file_record.get = AsyncMock(return_value=mock_file)
        
        instances = await detection_service.detect(
            file_id=ObjectId(),
            backend="test_backend",
            save=False
        )
        
        # Should default to zero bbox
        assert instances[0].bbox == {"x": 0, "y": 0, "w": 0, "h": 0}
