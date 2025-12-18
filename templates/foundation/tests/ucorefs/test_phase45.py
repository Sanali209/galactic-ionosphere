"""
UCoreFS Phase 4.5 Tests - Background AI Pipeline

Tests for:
- SimilarityService
- LLMService  
- AI task handlers
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId


class TestSimilarityService:
    """Tests for SimilarityService."""
    
    @pytest.fixture
    def mock_locator(self):
        locator = MagicMock()
        locator.get_system = MagicMock(return_value=AsyncMock())
        return locator
    
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.data = MagicMock()
        return config
    
    @pytest.mark.asyncio
    async def test_similarity_service_initialize(self, mock_locator, mock_config):
        """Test SimilarityService initialization."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        service = SimilarityService(mock_locator, mock_config)
        await service.initialize()
        
        assert service.is_ready == True
        assert service.default_threshold == 0.85
    
    @pytest.mark.asyncio
    async def test_find_and_create_relations_no_vector(self, mock_locator, mock_config):
        """Test similarity search for file without vector."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        from src.ucorefs.models.file_record import FileRecord
        
        service = SimilarityService(mock_locator, mock_config)
        await service.initialize()
        
        # Mock file without vector
        mock_file = FileRecord(has_vector=False)
        
        with patch('src.ucorefs.models.file_record.FileRecord.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_file
            
            count = await service.find_and_create_relations(ObjectId())
            
            assert count == 0


class TestLLMService:
    """Tests for LLMService."""
    
    @pytest.fixture
    def mock_locator(self):
        locator = MagicMock()
        locator.get_system = MagicMock(return_value=AsyncMock())
        return locator
    
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.data = MagicMock()
        return config
    
    @pytest.mark.asyncio
    async def test_llm_service_initialize(self, mock_locator, mock_config):
        """Test LLMService initialization."""
        from src.ucorefs.ai.llm_service import LLMService
        
        service = LLMService(mock_locator, mock_config)
        await service.initialize()
        
        assert service.is_ready == True
        assert service.llm_provider == "placeholder"
    
    @pytest.mark.asyncio
    async def test_generate_description_disabled(self, mock_locator, mock_config):
        """Test description generation when LLM is disabled."""
        from src.ucorefs.ai.llm_service import LLMService
        
        service = LLMService(mock_locator, mock_config)
        service.llm_enabled = False
        
        result = await service.generate_description(ObjectId())
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_batch_descriptions(self, mock_locator, mock_config):
        """Test batch description generation."""
        from src.ucorefs.ai.llm_service import LLMService
        
        service = LLMService(mock_locator, mock_config)
        await service.initialize()
        
        file_ids = [ObjectId(), ObjectId(), ObjectId()]
        
        results = await service.generate_batch_descriptions(file_ids, max_concurrent=2)
        
        assert isinstance(results, dict)


class TestTaskHandlers:
    """Tests for AI task handlers."""
    
    @pytest.mark.asyncio
    async def test_vectorize_clip_handler_file_not_found(self):
        """Test CLIP handler with missing file."""
        from src.ucorefs.ai.task_handlers import vectorize_clip_handler
        
        with patch('src.ucorefs.models.file_record.FileRecord.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            
            result = await vectorize_clip_handler(str(ObjectId()))
            
            assert result["success"] == False
            assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_vectorize_blip_handler(self):
        """Test BLIP caption handler."""
        from src.ucorefs.ai.task_handlers import vectorize_blip_handler
        from src.ucorefs.models.file_record import FileRecord
        
        mock_file = FileRecord(path="/test.jpg", extension="jpg")
        
        with patch('src.ucorefs.models.file_record.FileRecord.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_file
            
            with patch('src.ucorefs.types.registry.registry.get_driver') as mock_driver:
                mock_driver_instance = AsyncMock()
                mock_driver_instance.get_blip_caption = AsyncMock(return_value=None)
                mock_driver.return_value = mock_driver_instance
                
                result = await vectorize_blip_handler(str(ObjectId()))
                
                assert result["success"] == False
    
    def test_task_handlers_registry(self):
        """Test that all handlers are registered."""
        from src.ucorefs.ai.task_handlers import TASK_HANDLERS
        
        assert "vectorize_clip" in TASK_HANDLERS
        assert "vectorize_blip" in TASK_HANDLERS
        assert "find_similar" in TASK_HANDLERS
        assert "generate_description" in TASK_HANDLERS
