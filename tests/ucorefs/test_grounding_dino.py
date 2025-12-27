"""
UCoreFS - GroundingDINO Extractor Tests

Tests for GroundingDINOExtractor class mapping and detection storage.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId


class TestGroundingDINOExtractor:
    """Tests for GroundingDINOExtractor."""
    
    def test_grounding_dino_initialization(self):
        """Test GroundingDINOExtractor can be initialized."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        
        assert extractor.name == "grounding_dino"
        assert extractor.phase == 3
        assert extractor._store_as_instances == True
    
    def test_default_class_mapping(self):
        """Test default class mapping ontology."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        mapping = extractor.get_class_mapping()
        
        # Check key mappings
        assert "person" in mapping
        assert mapping["person"] == "person"
        assert "face" in mapping
        assert mapping["face"] == "face"
        assert "human face" in mapping
        assert mapping["human face"] == "face"  # Alternative phrase -> same class
        assert "car" in mapping
        assert mapping["car"] == "vehicle"  # Grouped under vehicle
    
    def test_custom_class_mapping(self):
        """Test custom class mapping via config."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        custom_mapping = {
            "human": "person",
            "dog face": "animal",
            "cat face": "animal",
        }
        
        config = {"class_mapping": custom_mapping}
        extractor = GroundingDINOExtractor(config=config)
        
        mapping = extractor.get_class_mapping()
        assert mapping == custom_mapping
        assert "human" in extractor._phrases
    
    def test_set_class_mapping(self):
        """Test updating class mapping at runtime."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        
        new_mapping = {
            "cat": "feline",
            "dog": "canine",
        }
        extractor.set_class_mapping(new_mapping)
        
        assert extractor.get_class_mapping() == new_mapping
        assert extractor._phrases == ["cat", "dog"]
    
    def test_phrases_derived_from_mapping(self):
        """Test that detection phrases come from mapping keys."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        config = {
            "class_mapping": {
                "person": "person",
                "face": "face",
            }
        }
        extractor = GroundingDINOExtractor(config=config)
        
        assert set(extractor._phrases) == {"person", "face"}
    
    def test_store_as_instances_config(self):
        """Test store_as_instances configuration."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        # Default: True
        extractor1 = GroundingDINOExtractor()
        assert extractor1._store_as_instances == True
        
        # Explicit: False
        config = {"store_as_instances": False}
        extractor2 = GroundingDINOExtractor(config=config)
        assert extractor2._store_as_instances == False
    
    def test_box_threshold_config(self):
        """Test box_threshold configuration."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        # Default
        extractor1 = GroundingDINOExtractor()
        assert extractor1._box_threshold == 0.35
        
        # Custom
        config = {"box_threshold": 0.5}
        extractor2 = GroundingDINOExtractor(config=config)
        assert extractor2._box_threshold == 0.5
    
    def test_can_process_images(self):
        """Test can_process filters by file type."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        
        # Image file - should process
        image_file = MagicMock()
        image_file.file_type = "image"
        assert extractor.can_process(image_file) == True
        
        # Text file - should not process
        text_file = MagicMock()
        text_file.file_type = "text"
        assert extractor.can_process(text_file) == False
    
    def test_grounding_dino_registered_in_phase3(self):
        """Test GroundingDINOExtractor is registered in Phase 3."""
        from src.ucorefs.extractors import ExtractorRegistry
        
        # Force import
        import src.ucorefs.extractors
        
        registered = ExtractorRegistry.list_registered()
        phase3_names = registered.get(3, [])
        
        assert "grounding_dino" in phase3_names
    
    def test_class_mapping_docstring(self):
        """Test class mapping is documented."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        # Check module docstring explains ontology
        import src.ucorefs.extractors.grounding_dino_extractor as module
        assert "class_mapping" in module.__doc__
        assert "ontology" in module.__doc__.lower()
    
    @pytest.mark.asyncio
    async def test_get_or_create_class(self):
        """Test _get_or_create_class creates DetectionClass."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        
        # Mock DetectionClass
        with patch('src.ucorefs.detection.models.DetectionClass') as MockClass:
            # First call - not found, create new
            MockClass.find_one = AsyncMock(return_value=None)
            MockClass.return_value.save = AsyncMock()
            
            await extractor._get_or_create_class("test_class")
            
            # Should have searched for existing
            MockClass.find_one.assert_called_once()
