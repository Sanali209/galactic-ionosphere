"""
UCoreFS Phase 4 Tests - Thumbnails & Vectors

Tests for:
- ThumbnailService caching and generation
- VectorService ChromaDB integration
- Hybrid search functionality
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
import tempfile
from pathlib import Path


class TestThumbnailService:
    """Tests for ThumbnailService."""
    
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
    async def test_thumbnail_service_initialize(self, mock_locator, mock_config):
        """Test ThumbnailService initialization."""
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        service = ThumbnailService(mock_locator, mock_config)
        await service.initialize()
        
        assert service.is_ready == True
        assert service.sizes == [128, 256, 512]
        assert service.cache_path.exists()
    
    def test_get_thumbnail_path(self, mock_locator, mock_config):
        """Test thumbnail path generation."""
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        service = ThumbnailService(mock_locator, mock_config)
        service.cache_path = Path("./thumbnails")
        
        file_id = ObjectId("507f1f77bcf86cd799439011")
        path = service.get_path(file_id, size=256)
        
        assert path is not None
        assert "507f1f77bcf86cd799439011_256.webp" in str(path)
    
    @pytest.mark.asyncio
    async def test_get_or_create_cached(self, mock_locator, mock_config):
        """Test getting cached thumbnail."""
        from src.ucorefs.thumbnails.service import ThumbnailService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ThumbnailService(mock_locator, mock_config)
            service.cache_path = Path(tmpdir)
            
            file_id = ObjectId()
            
            # Create fake cached thumbnail
            thumb_path = service.get_path(file_id, 256)
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(b"fake_thumbnail")
            
            # Get should return cached
            result = await service.get_or_create(file_id, 256)
            
            assert result == b"fake_thumbnail"


class TestVectorService:
    """Tests for VectorService."""
    
    @pytest.fixture
    def mock_locator(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_vector_service_initialize(self, mock_locator, mock_config):
        """Test VectorService initialization."""
        from src.ucorefs.vectors.service import VectorService
        
        service = VectorService(mock_locator, mock_config)
        
        # Initialize (may warn if ChromaDB not installed)
        await service.initialize()
        
        assert service.is_ready == True
    
    @pytest.mark.asyncio
    async def test_build_metadata_payload(self, mock_locator, mock_config):
        """Test metadata payload building."""
        from src.ucorefs.vectors.service import VectorService
        from src.ucorefs.models.file_record import FileRecord
        
        service = VectorService(mock_locator, mock_config)
        await service.initialize()
        
        # Mock file record
        file_id = ObjectId()
        mock_file = FileRecord(
            _id=file_id,
            path="/test/image.jpg",
            name="image.jpg",
            file_type="image",
            extension="jpg",
            label="red",
            rating=4
        )
        
        with patch('src.ucorefs.models.file_record.FileRecord.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_file
            
            payload = await service.build_metadata_payload(file_id)
            
            assert payload["file_type"] == "image"
            assert payload["extension"] == "jpg"
            assert payload["label"] == "red"
            assert payload["rating"] == 4
            assert payload["name"] == "image.jpg"
    
    @pytest.mark.asyncio
    async def test_upsert_when_unavailable(self, mock_locator, mock_config):
        """Test upsert when ChromaDB is unavailable."""
        from src.ucorefs.vectors.service import VectorService
        
        service = VectorService(mock_locator, mock_config)
        service._chroma_available = False
        
        result = await service.upsert(
            "file_embeddings",
            ObjectId(),
            [0.1] * 512,
            {}
        )
        
        assert result == False
    
    @pytest.mark.asyncio
    async def test_search_when_unavailable(self, mock_locator, mock_config):
        """Test search when ChromaDB is unavailable."""
        from src.ucorefs.vectors.service import VectorService
        
        service = VectorService(mock_locator, mock_config)
        service._chroma_available = False
        
        results = await service.search(
            "file_embeddings",
            [0.1] * 512
        )
        
        assert results == []


class TestHybridSearch:
    """Tests for hybrid search (vector + metadata)."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        """Test vector search with metadata filters."""
        from src.ucorefs.vectors.service import VectorService
        
        # This would require ChromaDB to be installed
        # For now, just test the interface
        
        service = VectorService(MagicMock(), MagicMock())
        await service.initialize()
        
        # Test search interface
        filters = {
            "file_type": "image",
            "rating": {"$gte": 3}
        }
        
        results = await service.search(
            "file_embeddings",
            [0.1] * 512,
            filters=filters,
            limit=10
        )
        
        # Will return empty if ChromaDB not available
        assert isinstance(results, list)
