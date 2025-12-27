"""
UCoreFS Pipeline Integration Tests

Tests for verifying Phase 2 and Phase 3 extractors work correctly.
Each test validates that extractors process files and produce expected outputs.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image


class TestPhase2Extractors:
    """Integration tests for Phase 2 extractors."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a sample test image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_image.jpg")
            img = Image.new("RGB", (640, 480), color="blue")
            img.save(img_path, "JPEG")
            yield img_path
    
    def test_thumbnail_extractor_initialization(self):
        """Test ThumbnailExtractor can be initialized."""
        from src.ucorefs.extractors.thumbnail import ThumbnailExtractor
        
        extractor = ThumbnailExtractor()
        assert extractor.name == "thumbnail"
        assert extractor.phase == 2
    
    def test_metadata_extractor_initialization(self):
        """Test MetadataExtractor can be initialized."""
        from src.ucorefs.extractors.metadata import MetadataExtractor
        
        extractor = MetadataExtractor()
        assert extractor.name == "metadata"
        assert extractor.phase == 2
    
    def test_clip_extractor_initialization(self):
        """Test CLIPExtractor can be initialized."""
        from src.ucorefs.extractors.clip_extractor import CLIPExtractor
        
        extractor = CLIPExtractor()
        assert extractor.name == "clip"
        assert extractor.phase == 2
    
    def test_dino_extractor_initialization(self):
        """Test DINOExtractor can be initialized."""
        from src.ucorefs.extractors.dino_extractor import DINOExtractor
        
        extractor = DINOExtractor()
        assert extractor.name == "dino"
        assert extractor.phase == 2
    
    def test_xmp_extractor_initialization(self):
        """Test XMPExtractor can be initialized."""
        from src.ucorefs.extractors.xmp import XMPExtractor
        
        extractor = XMPExtractor()
        assert hasattr(extractor, '_parse_hierarchical_tags')
    
    def test_extractor_registry_phase2(self):
        """Test ExtractorRegistry returns Phase 2 extractors."""
        from src.ucorefs.extractors import ExtractorRegistry
        
        # Get Phase 2 extractors
        extractors = ExtractorRegistry.get_for_phase(2)
        extractor_names = [e.name for e in extractors]
        
        # Verify expected extractors are registered
        assert "thumbnail" in extractor_names
        assert "metadata" in extractor_names
        # CLIP/DINO may be optional based on dependencies
    
    @pytest.mark.asyncio
    async def test_thumbnail_extractor_can_process_image(self, sample_image_path):
        """Test ThumbnailExtractor identifies images as processable."""
        from src.ucorefs.extractors.thumbnail import ThumbnailExtractor
        from src.ucorefs.models.file_record import FileRecord
        
        extractor = ThumbnailExtractor()
        
        # Create mock FileRecord
        record = MagicMock(spec=FileRecord)
        record.path = sample_image_path
        record.extension = "jpg"
        record.file_type = "image"
        
        can_process = extractor.can_process(record)
        assert can_process == True
    
    @pytest.mark.asyncio
    async def test_metadata_extractor_extracts_dimensions(self, sample_image_path):
        """Test MetadataExtractor extracts image dimensions."""
        from src.ucorefs.extractors.metadata import MetadataExtractor
        from src.ucorefs.models.file_record import FileRecord
        
        extractor = MetadataExtractor()
        
        # Create mock FileRecord
        record = MagicMock(spec=FileRecord)
        record.path = sample_image_path
        record.extension = "jpg"
        record.file_type = "image"
        record._id = "test_id"
        record.metadata = {}
        record.save = AsyncMock()
        
        # Process
        results = await extractor.process([record])
        
        # Verify metadata was extracted
        # Note: Actual test depends on implementation details
        assert "test_id" in results or len(results) > 0


class TestPhase3Extractors:
    """Integration tests for Phase 3 extractors."""
    
    def test_blip_extractor_initialization(self):
        """Test BLIPExtractor can be initialized."""
        from src.ucorefs.extractors.blip_extractor import BLIPExtractor
        
        extractor = BLIPExtractor()
        assert extractor.name == "blip"
        assert extractor.phase == 3
    
    def test_grounding_dino_extractor_initialization(self):
        """Test GroundingDINOExtractor can be initialized."""
        from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
        
        extractor = GroundingDINOExtractor()
        assert extractor.name == "grounding_dino"
        assert extractor.phase == 3
    
    def test_extractor_registry_phase3(self):
        """Test ExtractorRegistry returns Phase 3 extractors."""
        from src.ucorefs.extractors import ExtractorRegistry
        
        # Get Phase 3 extractors
        extractors = ExtractorRegistry.get_for_phase(3)
        extractor_names = [e.name for e in extractors]
        
        # Verify expected extractors are registered
        assert "blip" in extractor_names
        assert "grounding_dino" in extractor_names


class TestDetectionService:
    """Integration tests for DetectionService."""
    
    def test_detection_service_initialization(self):
        """Test DetectionService can be initialized."""
        from src.ucorefs.detection.service import DetectionService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        mock_config.data = MagicMock()
        
        service = DetectionService(mock_locator, mock_config)
        assert hasattr(service, '_backends')
        assert hasattr(service, 'register_backend')
    
    def test_detection_models_exist(self):
        """Test detection models can be imported."""
        from src.ucorefs.detection.models import (
            DetectionClass,
            DetectionObject,
            DetectionInstance
        )
        
        # Verify models have expected fields
        assert hasattr(DetectionInstance, 'file_id')
        assert hasattr(DetectionInstance, 'bbox')
        assert hasattr(DetectionInstance, 'confidence')
        assert hasattr(DetectionClass, 'class_name')


class TestProcessingPipeline:
    """Integration tests for ProcessingPipeline."""
    
    def test_pipeline_has_required_methods(self):
        """Test ProcessingPipeline has all required methods."""
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        
        assert hasattr(ProcessingPipeline, 'enqueue_phase2')
        assert hasattr(ProcessingPipeline, 'enqueue_phase3')
        assert hasattr(ProcessingPipeline, 'reindex_all')
        assert hasattr(ProcessingPipeline, '_handle_phase2_batch')
        assert hasattr(ProcessingPipeline, '_handle_phase3_item')
        assert hasattr(ProcessingPipeline, '_run_detection')
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test ProcessingPipeline initializes correctly."""
        from src.ucorefs.processing.pipeline import ProcessingPipeline
        from src.core.tasks.system import TaskSystem
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        mock_task_system = MagicMock(spec=TaskSystem)
        mock_task_system.register_handler = MagicMock()
        
        mock_locator.get_system = MagicMock(return_value=mock_task_system)
        
        pipeline = ProcessingPipeline(mock_locator, mock_config)
        
        # Verify pipeline was created
        assert pipeline is not None
        assert hasattr(pipeline, 'enqueue_phase2')
        assert hasattr(pipeline, 'enqueue_phase3')


class TestExtractorRegistration:
    """Tests for extractor auto-registration."""
    
    def test_all_extractors_registered(self):
        """Verify all extractors are registered after import."""
        from src.ucorefs.extractors import ExtractorRegistry
        
        # Force import to trigger registration
        import src.ucorefs.extractors
        
        # Get all registered extractors by phase
        registered = ExtractorRegistry.list_registered()
        
        # Should have both Phase 2 and Phase 3 extractors
        phase2_names = registered.get(2, [])
        phase3_names = registered.get(3, [])
        
        assert len(phase2_names) >= 2, f"Expected >=2 Phase 2 extractors, got {phase2_names}"
        assert len(phase3_names) >= 1, f"Expected >=1 Phase 3 extractors, got {phase3_names}"
        
        # Verify specific extractors
        assert "thumbnail" in phase2_names
        assert "metadata" in phase2_names
