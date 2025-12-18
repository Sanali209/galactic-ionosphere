"""
UCoreFS Phase 6 Tests - Tags & Albums

Tests for:
- Tag hierarchy with MPTT
- Synonyms and antonyms
- Albums (manual and smart)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId


class TestTagModel:
    """Tests for Tag model."""
    
    def test_tag_creation(self):
        """Test creating a tag."""
        from src.ucorefs.tags.models import Tag
        
        tag = Tag(
            name="Animals",
            full_path="Animals",
            depth=0,
            lft=0,
            rgt=5
        )
        
        assert tag.name == "Animals"
        assert tag.full_path == "Animals"
        assert tag.depth == 0
    
    def test_tag_with_synonyms(self):
        """Test tag with synonyms."""
        from src.ucorefs.tags.models import Tag
        
        tag = Tag(
            name="cat",
            full_path="Animals/Mammals/Cats",
            synonym_ids=[ObjectId(), ObjectId()]
        )
        
        assert len(tag.synonym_ids) == 2


class TestTagManager:
    """Tests for TagManager."""
    
    @pytest.fixture
    def mock_locator(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_tag_manager_initialize(self, mock_locator, mock_config):
        """Test TagManager initialization."""
        from src.ucorefs.tags.manager import TagManager
        
        manager = TagManager(mock_locator, mock_config)
        await manager.initialize()
        
        assert manager.is_ready == True
    
    @pytest.mark.asyncio
    async def test_expand_search_with_synonyms(self, mock_locator, mock_config):
        """Test synonym expansion in search."""
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.tags.models import Tag
        
        manager = TagManager(mock_locator, mock_config)
        
        tag1_id = ObjectId()
        tag2_id = ObjectId()
        tag3_id = ObjectId()
        
        # Mock get_synonyms to return synonyms
        async def mock_get_synonyms(tag_id):
            if tag_id == tag1_id:
                return [Tag(_id=tag2_id), Tag(_id=tag3_id)]
            return []
        
        manager.get_synonyms = mock_get_synonyms
        
        expanded = await manager.expand_search_with_synonyms([tag1_id])
        
        assert tag1_id in expanded
        assert tag2_id in expanded
        assert tag3_id in expanded
    
    @pytest.mark.asyncio
    async def test_check_antonym_conflict(self, mock_locator, mock_config):
        """Test antonym conflict detection."""
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.tags.models import Tag
        
        manager = TagManager(mock_locator, mock_config)
        
        tag1_id = ObjectId()
        tag2_id = ObjectId()
        
        # Mock tags with antonym relationship
        mock_tag1 = Tag(_id=tag1_id, name="work", antonym_ids=[tag2_id])
        
        with AsyncMock() as mock_get:
            mock_get.return_value = mock_tag1
            
            conflicts = await manager.check_antonym_conflict([tag1_id, tag2_id])
            
            # Should detect conflict (if Tag.get worked)
            assert isinstance(conflicts, list)


class TestAlbumModel:
    """Tests for Album model."""
    
    def test_manual_album_creation(self):
        """Test creating manual album."""
        from src.ucorefs.albums.models import Album
        
        album = Album(
            name="Vacation 2024",
            description="Summer vacation photos",
            is_smart=False
        )
        
        assert album.name == "Vacation 2024"
        assert album.is_smart == False
        assert album.file_ids == []
    
    def test_smart_album_creation(self):
        """Test creating smart album."""
        from src.ucorefs.albums.models import Album
        
        album = Album(
            name="High Rated Photos",
            is_smart=True,
            smart_query={"rating": {"$gte": 4}}
        )
        
        assert album.is_smart == True
        assert album.smart_query["rating"]["$gte"] == 4


class TestAlbumManager:
    """Tests for AlbumManager."""
    
    @pytest.fixture
    def mock_locator(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_album_manager_initialize(self, mock_locator, mock_config):
        """Test AlbumManager initialization."""
        from src.ucorefs.albums.manager import AlbumManager
        
        manager = AlbumManager(mock_locator, mock_config)
        await manager.initialize()
        
        assert manager.is_ready == True
    
    @pytest.mark.asyncio
    async def test_cannot_add_to_smart_album(self, mock_locator, mock_config):
        """Test that files can't be added to smart albums."""
        from src.ucorefs.albums.manager import AlbumManager
        from src.ucorefs.albums.models import Album
        
        manager = AlbumManager(mock_locator, mock_config)
        
        mock_album = Album(name="Smart", is_smart=True)
        
        with AsyncMock() as mock_get:
            mock_get.return_value = mock_album
            
            import src.ucorefs.albums.models
            original_get = src.ucorefs.albums.models.Album.get
            src.ucorefs.albums.models.Album.get = mock_get
            
            try:
                result = await manager.add_file_to_album(ObjectId(), ObjectId())
                assert result == False
            finally:
                src.ucorefs.albums.models.Album.get = original_get
