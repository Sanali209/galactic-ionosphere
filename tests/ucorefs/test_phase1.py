"""
UCoreFS Phase 1 Tests - Core Models and FSService

Tests for:
- FSRecord, FileRecord, DirectoryRecord CRUD
- FSService entry points API
- Virtual file creation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from datetime import datetime


class TestFSRecord:
    """Tests for FSRecord base model."""
    
    def test_fs_record_creation(self):
        """Test FSRecord can be created with default values."""
        from src.ucorefs.models.base import FSRecord
        
        record = FSRecord(path="/test/path", name="test")
        
        assert record.path == "/test/path"
        assert record.name == "test"
        assert record.is_virtual == False
        assert record.driver_type == "default"
        assert record.size_bytes == 0
    
    def test_fs_record_with_virtual_flag(self):
        """Test creating virtual FSRecord."""
        from src.ucorefs.models.base import FSRecord
        
        record = FSRecord(
            path="/virtual/item",
            name="virtual_item",
            is_virtual=True,
            driver_type="unity_package"
        )
        
        assert record.is_virtual == True
        assert record.driver_type == "unity_package"


class TestFileRecord:
    """Tests for FileRecord model."""
    
    def test_file_record_creation(self):
        """Test FileRecord creation with file-specific fields."""
        from src.ucorefs.models.file_record import FileRecord
        
        file = FileRecord(
            path="/photos/test.jpg",
            name="test.jpg",
            file_type="image",
            extension="jpg",
            size_bytes=1024
        )
        
        assert file.file_type == "image"
        assert file.extension == "jpg"
        assert file.size_bytes == 1024
        assert file.rating == 0
        assert file.favorite == False
    
    def test_file_record_metadata_fields(self):
        """Test FileRecord with metadata."""
        from src.ucorefs.models.file_record import FileRecord
        
        file = FileRecord(
            path="/test.jpg",
            name="test.jpg",
            rating=4,
            favorite=True,
            label="red",
            description="Test image"
        )
        
        assert file.rating == 4
        assert file.favorite == True
        assert file.label == "red"
        assert file.description == "Test image"


class TestDirectoryRecord:
    """Tests for DirectoryRecord model."""
    
    def test_directory_record_creation(self):
        """Test DirectoryRecord creation."""
        from src.ucorefs.models.directory import DirectoryRecord
        
        dir_record = DirectoryRecord(
            path="/photos",
            name="photos",
            child_count=10,
            file_count=100
        )
        
        assert dir_record.path == "/photos"
        assert dir_record.child_count == 10
        assert dir_record.file_count == 100
        assert dir_record.is_root == False
    
    def test_directory_as_library_root(self):
        """Test DirectoryRecord as library root."""
        from src.ucorefs.models.directory import DirectoryRecord
        
        root = DirectoryRecord(
            path="/media/photos",
            name="photos",
            is_root=True,
            watch_extensions=["jpg", "png"],
            blacklist_paths=["/media/photos/.cache"]
        )
        
        assert root.is_root == True
        assert root.watch_extensions == ["jpg", "png"]
        assert "/media/photos/.cache" in root.blacklist_paths


class TestFSService:
    """Tests for FSService entry points."""
    
    @pytest.fixture
    def mock_locator(self):
        """Create mock locator."""
        locator = MagicMock()
        locator.get_system = MagicMock(return_value=AsyncMock())
        return locator
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_fs_service_initialize(self, mock_locator, mock_config):
        """Test FSService initialization."""
        from src.core.services.fs_service import FSService
        
        service = FSService(mock_locator, mock_config)
        await service.initialize()
        
        assert service.is_ready == True
    
    @pytest.mark.asyncio
    @patch('src.ucorefs.models.directory.DirectoryRecord.find')
    async def test_get_roots(self, mock_find, mock_locator, mock_config):
        """Test getting library roots."""
        from src.core.services.fs_service import FSService
        from src.ucorefs.models.directory import DirectoryRecord
        
        mock_roots = [
            DirectoryRecord(path="/photos", name="photos", is_root=True),
            DirectoryRecord(path="/videos", name="videos", is_root=True)
        ]
        mock_find.return_value = mock_roots
        
        service = FSService(mock_locator, mock_config)
        await service.initialize()
        
        roots = await service.get_roots()
        
        assert len(roots) == 2
        mock_find.assert_called_with({"is_root": True})
    
    @pytest.mark.asyncio
    async def test_create_file_logs_zero_size(self, mock_locator, mock_config):
        """Test that zero-size files are logged."""
        from src.core.services.fs_service import FSService
        
        service = FSService(mock_locator, mock_config)
        service._journal = AsyncMock()
        
        with patch('src.ucorefs.models.file_record.FileRecord.save', new_callable=AsyncMock):
            file = await service.create_file(
                path="/empty.txt",
                name="empty.txt",
                size_bytes=0
            )
        
        # Verify journal was called for zero-size warning
        service._journal.log.assert_called()


class TestVirtualEntries:
    """Tests for virtual file/directory creation."""
    
    def test_virtual_file_creation(self):
        """Test creating virtual file entries."""
        from src.ucorefs.models.file_record import FileRecord
        
        virtual = FileRecord(
            path="unity://package/asset.fbx",
            name="asset.fbx",
            is_virtual=True,
            driver_type="unity_package",
            file_type="3d"
        )
        
        assert virtual.is_virtual == True
        assert virtual.driver_type == "unity_package"
    
    def test_virtual_directory_creation(self):
        """Test creating virtual directory entries."""
        from src.ucorefs.models.directory import DirectoryRecord
        
        virtual_dir = DirectoryRecord(
            path="archive://test.zip/folder",
            name="folder",
            is_virtual=True,
            driver_type="archive"
        )
        
        assert virtual_dir.is_virtual == True
        assert virtual_dir.driver_type == "archive"
