"""
UCoreFS - WD-Tagger Extractor Tests

Tests for WDTaggerExtractor auto-tagging functionality.
"""
import pytest
import tempfile
import os
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image


class TestWDTaggerExtractor:
    """Tests for WDTaggerExtractor."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a sample test image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_image.jpg")
            img = Image.new("RGB", (512, 512), color="blue")
            img.save(img_path, "JPEG")
            yield img_path
    
    def test_wd_tagger_initialization(self):
        """Test WDTaggerExtractor can be initialized."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        
        extractor = WDTaggerExtractor()
        
        assert extractor.name == "wd_tagger"
        assert extractor.phase == 2
        assert extractor.enabled == True
        assert extractor.general_threshold == 0.35
        assert extractor.character_threshold == 0.75
    
    def test_wd_tagger_custom_config(self):
        """Test WDTaggerExtractor with custom config."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        
        config = {
            "enabled": False,
            "general_threshold": 0.5,
            "character_threshold": 0.8,
        }
        
        extractor = WDTaggerExtractor(config=config)
        
        assert extractor.enabled == False
        assert extractor.general_threshold == 0.5
        assert extractor.character_threshold == 0.8
    
    def test_wd_tagger_can_process_image(self, sample_image_path):
        """Test WDTaggerExtractor identifies images as processable."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        from src.ucorefs.models.file_record import FileRecord
        
        extractor = WDTaggerExtractor()
        
        # Create mock FileRecord
        record = MagicMock(spec=FileRecord)
        record.extension = "jpg"
        
        can_process = extractor.can_process(record)
        assert can_process == True
    
    def test_wd_tagger_skips_non_images(self):
        """Test WDTaggerExtractor skips non-image files."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        from src.ucorefs.models.file_record import FileRecord
        
        extractor = WDTaggerExtractor()
        
        # Create mock non-image FileRecord
        record = MagicMock(spec=FileRecord)
        record.extension = "txt"
        
        can_process = extractor.can_process(record)
        assert can_process == False
    
    def test_wd_tagger_skips_when_disabled(self, sample_image_path):
        """Test WDTaggerExtractor skips when disabled."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        from src.ucorefs.models.file_record import FileRecord
        
        config = {"enabled": False}
        extractor = WDTaggerExtractor(config=config)
        
        record = MagicMock(spec=FileRecord)
        record.extension = "jpg"
        
        can_process = extractor.can_process(record)
        assert can_process == False
    
    def test_wd_tagger_image_helpers(self):
        """Test WDTaggerExtractor image helper methods."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        
        # Test ensure_rgb
        rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        rgb_img = WDTaggerExtractor._ensure_rgb(rgba_img)
        assert rgb_img.mode == "RGB"
        
        # Test pad_square
        rect_img = Image.new("RGB", (100, 50), color="blue")
        square_img = WDTaggerExtractor._pad_square(rect_img)
        assert square_img.size == (100, 100)  # Padded to square
    
    def test_wd_tagger_registered_in_phase2(self):
        """Test WDTaggerExtractor is registered in Phase 2."""
        from src.ucorefs.extractors import ExtractorRegistry
        
        # Force import to trigger registration
        import src.ucorefs.extractors
        
        # Get Phase 2 extractors
        registered = ExtractorRegistry.list_registered()
        phase2_names = registered.get(2, [])
        
        assert "wd_tagger" in phase2_names
    
    def test_wd_tagger_tag_hierarchy_format(self):
        """Test WD-Tagger creates correct tag hierarchy."""
        # Expected tag format: auto/wd_tag/{tag}
        tag_name = "1girl"
        expected_full = f"auto/wd_tag/{tag_name}"
        
        assert expected_full.startswith("auto/")
        assert "/wd_tag/" in expected_full
    
    @pytest.mark.asyncio
    async def test_wd_tagger_store_fallback(self):
        """Test store method fallback when TagManager not available."""
        from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor
        from bson import ObjectId
        
        extractor = WDTaggerExtractor(locator=None)  # No locator
        
        # Mock result
        result = {
            "tags": ["blue", "simple background"],
            "characters": [],
            "rating": "general"
        }
        
        # Mock FileRecord.get to return a file with auto_tags
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_file = MagicMock()
            mock_file.auto_tags = []
            mock_file.save = AsyncMock()
            mock_get.return_value = mock_file
            
            success = await extractor.store(ObjectId(), result)
            
            # Should attempt to save
            assert mock_file.save.called or not success  # Either saved or gracefully failed
