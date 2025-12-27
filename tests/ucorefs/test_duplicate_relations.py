"""
UCoreFS - Duplicate Relations Tests

Tests for SimilarityService duplicate marking functionality.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId


class TestSimilarityServiceDuplicates:
    """Tests for SimilarityService duplicate marking."""
    
    def test_similarity_service_has_mark_as_duplicate(self):
        """Test SimilarityService has mark_as_duplicate method."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        assert hasattr(SimilarityService, 'mark_as_duplicate')
    
    def test_similarity_service_has_get_duplicates(self):
        """Test SimilarityService has get_duplicates method."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        assert hasattr(SimilarityService, 'get_duplicates')
    
    def test_similarity_service_has_is_duplicate(self):
        """Test SimilarityService has is_duplicate method."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        assert hasattr(SimilarityService, 'is_duplicate')
    
    def test_duplicate_types_documented(self):
        """Test duplicate types are documented."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        docstring = SimilarityService.mark_as_duplicate.__doc__
        
        assert "exact_duplicate" in docstring
        assert "near_duplicate" in docstring
        assert "similar" in docstring
        assert "same_set" in docstring
    
    @pytest.mark.asyncio
    async def test_mark_as_duplicate_returns_result_dict(self):
        """Test mark_as_duplicate returns structured result."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        # Create service directly without initialize
        service = SimilarityService(mock_locator, mock_config)
        
        # Mock RelationService
        mock_relation_service = MagicMock()
        mock_relation_service.create_relation = AsyncMock(return_value=MagicMock(id=ObjectId()))
        mock_locator.get_system = MagicMock(return_value=mock_relation_service)
        
        file_id_1 = ObjectId()
        file_id_2 = ObjectId()
        
        result = await service.mark_as_duplicate(file_id_1, file_id_2, "near_duplicate")
        
        # Check result structure
        assert "success" in result
        assert "file_id_1" in result
        assert "file_id_2" in result
        assert "duplicate_type" in result
        assert result["duplicate_type"] == "near_duplicate"
    
    @pytest.mark.asyncio
    async def test_mark_as_duplicate_uses_bidirectional(self):
        """Test mark_as_duplicate creates bidirectional relation."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = SimilarityService(mock_locator, mock_config)
        
        # Mock RelationService
        mock_relation_service = MagicMock()
        mock_relation_service.create_relation = AsyncMock(return_value=MagicMock(id=ObjectId()))
        mock_locator.get_system = MagicMock(return_value=mock_relation_service)
        
        file_id_1 = ObjectId()
        file_id_2 = ObjectId()
        
        await service.mark_as_duplicate(file_id_1, file_id_2)
        
        # Check bidirectional=True was passed
        call_args = mock_relation_service.create_relation.call_args
        assert call_args is not None
        # The call should include bidirectional=True
        _, kwargs = call_args
        assert kwargs.get("bidirectional") == True
    
    @pytest.mark.asyncio
    async def test_get_duplicates_handles_empty(self):
        """Test get_duplicates returns empty list gracefully."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = SimilarityService(mock_locator, mock_config)
        
        # Mock RelationService with empty results
        mock_relation_service = MagicMock()
        mock_relation_service.get_relations = AsyncMock(return_value=[])
        mock_locator.get_system = MagicMock(return_value=mock_relation_service)
        
        result = await service.get_duplicates(ObjectId())
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_is_duplicate_returns_bool(self):
        """Test is_duplicate returns boolean."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = SimilarityService(mock_locator, mock_config)
        
        # Mock RelationService
        mock_relation_service = MagicMock()
        mock_relation_service.relation_exists = AsyncMock(return_value=False)
        mock_locator.get_system = MagicMock(return_value=mock_relation_service)
        
        result = await service.is_duplicate(ObjectId(), ObjectId())
        
        assert isinstance(result, bool)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_is_duplicate_checks_both_directions(self):
        """Test is_duplicate checks bidirectional relation."""
        from src.ucorefs.ai.similarity_service import SimilarityService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = SimilarityService(mock_locator, mock_config)
        
        # Mock RelationService - first check fails, second succeeds
        mock_relation_service = MagicMock()
        mock_relation_service.relation_exists = AsyncMock(side_effect=[False, True])
        mock_locator.get_system = MagicMock(return_value=mock_relation_service)
        
        result = await service.is_duplicate(ObjectId(), ObjectId())
        
        # Should be True because second direction exists
        assert result == True
        # Should have been called twice
        assert mock_relation_service.relation_exists.call_count == 2
