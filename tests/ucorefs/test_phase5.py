"""
UCoreFS Phase 5 Tests - Detection & Relations

Tests for:
- Detection models (DetectionClass, DetectionObject, DetectionInstance)
- Relation models and service
- Virtual detection instances
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId


class TestDetectionModels:
    """Tests for detection models."""
    
    def test_detection_class_creation(self):
        """Test creating detection class."""
        from src.ucorefs.detection.models import DetectionClass
        
        det_class = DetectionClass(
            class_name="face",
            path="/detections/face",
            name="face"
        )
        
        assert det_class.class_name == "face"
        assert det_class.lft == 0
        assert det_class.rgt == 0
    
    def test_detection_object_creation(self):
        """Test creating detection object."""
        from src.ucorefs.detection.models import DetectionObject
        
        det_obj = DetectionObject(
            object_name="John Smith",
            path="/detections/faces/john_smith",
            name="john_smith"
        )
        
        assert det_obj.object_name == "John Smith"
    
    def test_detection_instance_is_virtual(self):
        """Test that detection instance is virtual by default."""
        from src.ucorefs.detection.models import DetectionInstance
        
        instance = DetectionInstance(
            path="/virtual/detection/1",
            name="detection_1",
            bbox={"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.6}
        )
        
        assert instance.is_virtual == True
        assert instance.driver_type == "detection"
        assert instance.bbox["x"] == 0.1


class TestRelationModels:
    """Tests for relation models."""
    
    def test_relation_type_creation(self):
        """Test creating relation type."""
        from src.ucorefs.relations.models import RelationType
        
        rel_type = RelationType(
            type_name="image-image",
            description="Image to image",
            subtypes=["duplicate", "near_duplicate", "variant"]
        )
        
        assert rel_type.type_name == "image-image"
        assert "duplicate" in rel_type.subtypes
    
    def test_relation_creation(self):
        """Test creating relation."""
        from src.ucorefs.relations.models import Relation
        
        relation = Relation(
            source_id=ObjectId(),
            target_id=ObjectId(),
            relation_type="image-image",
            subtype="near_duplicate",
            payload={"score": 0.92, "threshold": 0.85}
        )
        
        assert relation.relation_type == "image-image"
        assert relation.subtype == "near_duplicate"
        assert relation.is_valid == True
        assert relation.payload["score"] == 0.92


class TestRelationService:
    """Tests for RelationService."""
    
    @pytest.fixture
    def mock_locator(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_relation_service_initialize(self, mock_locator, mock_config):
        """Test RelationService initialization."""
        from src.ucorefs.relations.service import RelationService
        
        with AsyncMock() as mock_init:
            service = RelationService(mock_locator, mock_config)
            service._init_default_types = mock_init
            
            await service.initialize()
            
            assert service.is_ready == True
    
    @pytest.mark.asyncio
    async def test_mark_wrong_relation(self, mock_locator, mock_config):
        """Test marking relation as wrong."""
        from src.ucorefs.relations.service import RelationService
        from src.ucorefs.relations.models import Relation
        
        service = RelationService(mock_locator, mock_config)
        
        mock_relation = Relation(
            source_id=ObjectId(),
            target_id=ObjectId(),
            relation_type="image-image",
            subtype="duplicate",
            is_valid=True
        )
        
        mock_relation.save = AsyncMock()
        
        with AsyncMock() as mock_get:
            mock_get.return_value = mock_relation
            
            # Mock Relation.get
            import src.ucorefs.relations.models
            original_get = src.ucorefs.relations.models.Relation.get
            src.ucorefs.relations.models.Relation.get = mock_get
            
            try:
                result = await service.mark_wrong(ObjectId())
                
                assert result == True
                assert mock_relation.is_valid == False
                assert mock_relation.subtype == "wrong"
            finally:
                src.ucorefs.relations.models.Relation.get = original_get


class TestVirtualDetections:
    """Tests for virtual detection instances."""
    
    def test_detection_as_virtual_file(self):
        """Test that detections behave as virtual files."""
        from src.ucorefs.detection.models import DetectionInstance
        
        # Detection is a virtual FSRecord
        detection = DetectionInstance(
            path="/image.jpg#detection1",
            name="person_face",
            file_id=ObjectId(),
            bbox={"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.5},
            confidence=0.95
        )
        
        # Should be searchable/filterable like regular files
        assert detection.is_virtual == True
        assert detection.path is not None
        assert detection.name is not None
        assert detection.confidence == 0.95
