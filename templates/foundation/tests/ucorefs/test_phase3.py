"""
UCoreFS Phase 3 Tests - File Types & Virtual Drivers

Tests for:
- IFileDriver interface
- FileTypeRegistry
- ImageDriver with XMP
- TextDriver
- Type-specific metadata extraction
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path


class TestFileTypeRegistry:
    """Tests for FileTypeRegistry."""
    
    def test_registry_registration(self):
        """Test driver registration."""
        from src.ucorefs.types.registry import FileTypeRegistry
        from src.ucorefs.types.default import DefaultDriver
        
        registry = FileTypeRegistry()
        registry.register(DefaultDriver, extensions=["test"])
        
        assert "test" in registry.list_supported_extensions()
    
    def test_get_driver_by_extension(self):
        """Test getting driver by extension."""
        from src.ucorefs.types.registry import FileTypeRegistry
        from src.ucorefs.types.image import ImageDriver
        
        registry = FileTypeRegistry()
        registry.register(ImageDriver)
        
        driver = registry.get_driver(extension="jpg")
        assert driver.driver_id == "image"
    
    def test_get_driver_by_path(self):
        """Test getting driver by file path."""
        from src.ucorefs.types.registry import FileTypeRegistry
        from src.ucorefs.types.image import ImageDriver
        
        registry = FileTypeRegistry()
        registry.register(ImageDriver)
        
        driver = registry.get_driver(path="/test/photo.jpg")
        assert driver.driver_id == "image"
    
    def test_get_default_driver_for_unknown(self):
        """Test fallback to default driver."""
        from src.ucorefs.types.registry import FileTypeRegistry
        
        registry = FileTypeRegistry()
        
        driver = registry.get_driver(extension="unknown")
        assert driver.driver_id == "default"


class TestXMPExtractor:
    """Tests for XMP metadata extraction."""
    
    def test_parse_hierarchical_tags_pipe_separator(self):
        """Test parsing tags with pipe separator."""
        from src.ucorefs.extractors.xmp import XMPExtractor
        
        extractor = XMPExtractor()
        
        raw_tags = ["Animals|Mammals|Cats"]
        result = extractor._parse_hierarchical_tags(raw_tags)
        
        assert "Animals" in result
        assert "Animals|Mammals" in result
        assert "Animals|Mammals|Cats" in result
    
    def test_parse_hierarchical_tags_slash_separator(self):
        """Test parsing tags with slash separator."""
        from src.ucorefs.extractors.xmp import XMPExtractor
        
        extractor = XMPExtractor()
        
        raw_tags = ["Nature/Landscapes/Mountains"]
        result = extractor._parse_hierarchical_tags(raw_tags)
        
        assert "Nature" in result
        assert "Nature/Landscapes" in result
        assert "Nature/Landscapes/Mountains" in result
    
    def test_parse_non_hierarchical_tags(self):
        """Test parsing flat tags."""
        from src.ucorefs.extractors.xmp import XMPExtractor
        
        extractor = XMPExtractor()
        
        raw_tags = ["vacation", "summer", "2024"]
        result = extractor._parse_hierarchical_tags(raw_tags)
        
        assert "vacation" in result
        assert "summer" in result
        assert "2024" in result


class TestImageDriver:
    """Tests for ImageDriver."""
    
    def test_can_handle_image_extensions(self):
        """Test image extension detection."""
        from src.ucorefs.types.image import ImageDriver
        
        driver = ImageDriver()
        
        assert driver.can_handle("/test/photo.jpg") == True
        assert driver.can_handle("/test/image.png") == True
        assert driver.can_handle("/test/doc.txt") == False
    
    @pytest.mark.asyncio
    async def test_extract_metadata_basic(self):
        """Test basic image metadata extraction."""
        from src.ucorefs.types.image import ImageDriver
        from src.ucorefs.models.file_record import FileRecord
        
        driver = ImageDriver()
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            from PIL import Image
            img = Image.new("RGB", (100, 50), color="red")
            img.save(tmp.name)
            
            try:
                record = FileRecord(
                    path=tmp.name,
                    name="test.png",
                    extension="png"
                )
                
                metadata = await driver.extract_metadata(record)
                
                assert metadata["file_type"] == "image"
                assert metadata["width"] == 100
                assert metadata["height"] == 50
            finally:
                os.unlink(tmp.name)


class TestTextDriver:
    """Tests for TextDriver."""
    
    def test_can_handle_text_extensions(self):
        """Test text extension detection."""
        from src.ucorefs.types.text import TextDriver
        
        driver = TextDriver()
        
        assert driver.can_handle("/test/file.txt") == True
        assert driver.can_handle("/test/readme.md") == True
        assert driver.can_handle("/test/image.jpg") == False
    
    @pytest.mark.asyncio
    async def test_extract_text_metadata(self):
        """Test text metadata extraction."""
        from src.ucorefs.types.text import TextDriver
        from src.ucorefs.models.file_record import FileRecord
        
        driver = TextDriver()
        
        # Create test text file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as tmp:
            tmp.write("Line 1\nLine 2\nLine 3\n")
            tmp.flush()
            
            try:
                record = FileRecord(
                    path=tmp.name,
                    name="test.txt",
                    extension="txt"
                )
                
                metadata = await driver.extract_metadata(record)
                
                assert metadata["file_type"] == "text"
                assert metadata["line_count"] == 4  # 3 lines + final newline
                assert metadata["encoding"] in ["utf-8", "latin-1"]
            finally:
                os.unlink(tmp.name)
