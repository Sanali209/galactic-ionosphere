"""
Unit tests for DetectionInstance model enhancements.
"""
import pytest
from unittest.mock import AsyncMock, patch
from bson import ObjectId

from src.ucorefs.detection.models import DetectionInstance, DetectionClass


@pytest.mark.asyncio
async def test_class_name_property_fallback_to_group_name():
    """Test that class_name property falls back to group_name when class not resolved."""
    instance = DetectionInstance(
        name="test_detection",
        file_id=ObjectId(),
        group_name="sedan",
        bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        confidence=0.95
    )
    
    # No class resolved yet, should return group_name
    assert instance.class_name == "sedan"


@pytest.mark.asyncio
async def test_class_name_property_returns_cached():
    """Test that class_name property returns cached value."""
    instance = DetectionInstance(
        name="test_detection",
        file_id=ObjectId(),
        group_name="car",
        bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        confidence=0.95
    )
    
    # Manually set cache
    instance._class_name_cache = "vehicle"
    
    # Should return cached value
    assert instance.class_name == "vehicle"


@pytest.mark.asyncio
async def test_resolve_class_name_caches_result():
    """Test that resolve_class_name() fetches and caches the class name."""
    class_id = ObjectId()
    instance = DetectionInstance(
        name="test_detection",
        file_id=ObjectId(),
        detection_class_id=class_id,
        group_name="car",
        bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        confidence=0.95
    )
    
    # Mock DetectionClass.get()
    mock_class = DetectionClass(
        name="vehicle_class",
        class_name="vehicle"
    )
    
    with patch.object(DetectionClass, 'get', new=AsyncMock(return_value=mock_class)):
        resolved_name = await instance.resolve_class_name()
        
        # Should fetch and cache
        assert resolved_name == "vehicle"
        assert instance._class_name_cache == "vehicle"
        
        # Subsequent access should use cache
        assert instance.class_name == "vehicle"


@pytest.mark.asyncio
async def test_resolve_class_name_fallback_when_no_class_id():
    """Test that resolve_class_name() falls back to group_name when no class_id."""
    instance = DetectionInstance(
        name="test_detection",
        file_id=ObjectId(),
        detection_class_id=None,  # No class ID
        group_name="unknown_object",
        bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        confidence=0.85
    )
    
    resolved_name = await instance.resolve_class_name()
    
    # Should fall back to group_name
    assert resolved_name == "unknown_object"
    assert instance._class_name_cache == "unknown_object"


@pytest.mark.asyncio
async def test_resolve_class_name_returns_cached_on_second_call():
    """Test that resolve_class_name() doesn't fetch again if already cached."""
    class_id = ObjectId()
    instance = DetectionInstance(
        name="test_detection",
        file_id=ObjectId(),
        detection_class_id=class_id,
        group_name="car",
        bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        confidence=0.95
    )
    
    mock_class = DetectionClass(
        name="vehicle_class",
        class_name="vehicle"
    )
    
    with patch.object(DetectionClass, 'get', new=AsyncMock(return_value=mock_class)) as mock_get:
        # First call
        await instance.resolve_class_name()
        assert mock_get.call_count == 1
        
        # Second call should use cache, not call get() again
        await instance.resolve_class_name()
        assert mock_get.call_count == 1  # Still 1, not 2
