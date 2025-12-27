"""
UCoreFS - Tag Enhancements Tests

Tests for TagManager get_tag_statistics and bulk_rename methods.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId


class TestTagManagerEnhancements:
    """Tests for TagManager enhanced methods."""
    
    def test_tag_manager_has_get_tag_statistics(self):
        """Test TagManager has get_tag_statistics method."""
        from src.ucorefs.tags.manager import TagManager
        
        assert hasattr(TagManager, 'get_tag_statistics')
    
    def test_tag_manager_has_bulk_rename(self):
        """Test TagManager has bulk_rename method."""
        from src.ucorefs.tags.manager import TagManager
        
        assert hasattr(TagManager, 'bulk_rename')
    
    def test_tag_manager_has_get_tags_report(self):
        """Test TagManager has get_tags_report method."""
        from src.ucorefs.tags.manager import TagManager
        
        assert hasattr(TagManager, 'get_tags_report')
    
    @pytest.mark.asyncio
    async def test_get_tag_statistics_empty(self):
        """Test get_tag_statistics with no tags."""
        from src.ucorefs.tags.manager import TagManager
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        manager = TagManager(mock_locator, mock_config)
        
        # Mock Tag.find to return empty list
        with patch('src.ucorefs.tags.models.Tag.find') as mock_find:
            mock_find.return_value = []
            
            stats = await manager.get_tag_statistics()
            
            assert stats["total_count"] == 0
            assert stats["root_count"] == 0
            assert stats["max_depth"] == 0
    
    @pytest.mark.asyncio
    async def test_get_tag_statistics_with_tags(self):
        """Test get_tag_statistics with sample tags."""
        from src.ucorefs.tags.manager import TagManager
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        manager = TagManager(mock_locator, mock_config)
        
        # Create mock tags
        mock_tag1 = MagicMock()
        mock_tag1.depth = 0
        mock_tag1.parent_id = None
        mock_tag1.full_path = "auto"
        
        mock_tag2 = MagicMock()
        mock_tag2.depth = 1
        mock_tag2.parent_id = ObjectId()
        mock_tag2.full_path = "auto/tag1"
        
        mock_tag3 = MagicMock()
        mock_tag3.depth = 1
        mock_tag3.parent_id = ObjectId()
        mock_tag3.full_path = "auto/tag2"
        
        with patch('src.ucorefs.tags.models.Tag.find') as mock_find:
            mock_find.return_value = [mock_tag1, mock_tag2, mock_tag3]
            
            stats = await manager.get_tag_statistics()
            
            assert stats["total_count"] == 3
            assert stats["root_count"] == 1
            assert stats["max_depth"] == 1
            assert stats["by_prefix"]["auto"] == 3
    
    @pytest.mark.asyncio
    async def test_bulk_rename_no_matches(self):
        """Test bulk_rename with no matching tags."""
        from src.ucorefs.tags.manager import TagManager
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        manager = TagManager(mock_locator, mock_config)
        
        with patch('src.ucorefs.tags.models.Tag.find') as mock_find:
            mock_find.return_value = []
            
            result = await manager.bulk_rename("old_prefix", "new_prefix")
            
            assert result["success"] == True
            assert result["renamed_count"] == 0
    
    @pytest.mark.asyncio
    async def test_bulk_rename_with_matches(self):
        """Test bulk_rename renames matching tags."""
        from src.ucorefs.tags.manager import TagManager
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        manager = TagManager(mock_locator, mock_config)
        
        # Create mock tag
        mock_tag = MagicMock()
        mock_tag.full_path = "auto/wd_tag/1girl"
        mock_tag.save = AsyncMock()
        
        with patch('src.ucorefs.tags.models.Tag.find') as mock_find:
            mock_find.return_value = [mock_tag]
            
            result = await manager.bulk_rename("auto/wd_tag", "generated/tags")
            
            assert result["success"] == True
            assert result["renamed_count"] == 1
            assert mock_tag.full_path == "generated/tags/1girl"
    
    @pytest.mark.asyncio
    async def test_get_tags_report_format(self):
        """Test get_tags_report returns formatted string."""
        from src.ucorefs.tags.manager import TagManager
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        manager = TagManager(mock_locator, mock_config)
        
        with patch('src.ucorefs.tags.models.Tag.find') as mock_find:
            mock_find.return_value = []
            
            report = await manager.get_tags_report()
            
            assert isinstance(report, str)
            assert "Tag Report" in report
            assert "Total Tags:" in report
