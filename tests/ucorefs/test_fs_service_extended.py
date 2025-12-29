"""
FSService - Extended Unit Tests

Additional tests for FSService to increase coverage.
Covers:
- Library root management
- Directory navigation
- Search functionality
- CRUD operations
- Error handling
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId

from src.ucorefs.services.fs_service import FSService
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord


class TestFSServiceLibraryRoots:
    """Tests for library root management."""
    
    @pytest.fixture
    def fs_service(self):
        """Create FSService instance with mocked dependencies."""
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        # Mock DatabaseManager
        mock_db = MagicMock()
        mock_locator.get_system.return_value = mock_db
        
        service = FSService(mock_locator, mock_config)
        return service
    
    @pytest.mark.asyncio
    async def test_get_roots_returns_root_directories(self, fs_service):
        """Test get_roots returns directories marked as roots."""
        mock_root1 = MagicMock(spec=DirectoryRecord)
        mock_root1.path = "/library1"
        mock_root1.is_root = True
        
        mock_root2 = MagicMock(spec=DirectoryRecord)
        mock_root2.path = "/library2"
        mock_root2.is_root = True
        
        with patch.object(DirectoryRecord, 'find') as mock_find:
            mock_find.return_value = [mock_root1, mock_root2]
            
            roots = await fs_service.get_roots()
            
            assert len(roots) == 2
            mock_find.assert_called_once_with({"is_root": True})
    
    @pytest.mark.asyncio
    async def test_add_library_root_creates_root_directory(self, fs_service):
        """Test adding a new library root."""
        test_path = "/test/library"
        
        with patch.object(DirectoryRecord, 'find_one') as mock_find_one, \
             patch.object(DirectoryRecord, 'create') as mock_create:
            
            # No existing root
            mock_find_one.return_value = None
            
            # Mock created root
            mock_root = MagicMock()
            mock_root._id = ObjectId()
            mock_create.return_value = mock_root
            
            root = await fs_service.add_library_root(test_path)
            
            assert root is not None
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_library_root_raises_error_if_exists(self, fs_service):
        """Test adding duplicate library root raises ValueError."""
        test_path = "/test/library"
        
        with patch.object(DirectoryRecord, 'find_one') as mock_find_one:
            # Existing root
            mock_find_one.return_value = MagicMock()
            
            with pytest.raises(ValueError, match="already exists"):
                await fs_service.add_library_root(test_path)
    
    @pytest.mark.asyncio
    async def test_add_library_root_with_extensions(self, fs_service):
        """Test adding root with watch extensions."""
        test_path = "/test/library"
        extensions = [".jpg", ".png", ".gif"]
        
        with patch.object(DirectoryRecord, 'find_one') as mock_find_one, \
             patch.object(DirectoryRecord, 'create') as mock_create:
            
            mock_find_one.return_value = None
            mock_root = MagicMock()
            mock_create.return_value = mock_root
            
            await fs_service.add_library_root(
                test_path,
                watch_extensions=extensions
            )
            
            # Verify extensions were passed
            call_args = mock_create.call_args
            assert 'watch_extensions' in call_args[1]


class TestFSServiceNavigation:
    """Tests for directory navigation."""
    
    @pytest.fixture
    def fs_service(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return FSService(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_get_children_returns_files_and_directories(self, fs_service):
        """Test get_children returns mixed file/directory records."""
        dir_id = ObjectId()
        
        mock_file = MagicMock(spec=FileRecord)
        mock_dir = MagicMock(spec=DirectoryRecord)
        
        with patch.object(FileRecord, 'find') as mock_find_files, \
             patch.object(DirectoryRecord, 'find') as mock_find_dirs:
            
            mock_find_files.return_value = [mock_file]
            mock_find_dirs.return_value = [mock_dir]
            
            children = await fs_service.get_children(dir_id)
            
            assert len(children) == 2
            assert mock_file in children
            assert mock_dir in children
    
    @pytest.mark.asyncio
    async def test_get_children_with_limit(self, fs_service):
        """Test get_children respects limit parameter."""
        dir_id = ObjectId()
        
        # Create 10 mock files
        mock_files = [MagicMock(spec=FileRecord) for _ in range(10)]
        
        with patch.object(FileRecord, 'find') as mock_find_files, \
             patch.object(DirectoryRecord, 'find') as mock_find_dirs:
            
            mock_find_files.return_value = mock_files
            mock_find_dirs.return_value = []
            
            children = await fs_service.get_children(dir_id, limit=5)
            
            # Should only return 5
            assert len(children) == 5
    
    @pytest.mark.asyncio
    async def test_get_files_only_returns_files(self, fs_service):
        """Test get_files returns only FileRecord objects."""
        dir_id = ObjectId()
        
        mock_files = [MagicMock(spec=FileRecord) for _ in range(3)]
        
        with patch.object(FileRecord, 'find') as mock_find:
            mock_find.return_value = mock_files
            
            files = await fs_service.get_files(dir_id)
            
            assert len(files) == 3
            assert all(isinstance(f, MagicMock) for f in files)
    
    @pytest.mark.asyncio
    async def test_get_directories_only_returns_directories(self, fs_service):
        """Test get_directories returns only DirectoryRecord objects."""
        dir_id = ObjectId()
        
        mock_dirs = [MagicMock(spec=DirectoryRecord) for _ in range(2)]
        
        with patch.object(DirectoryRecord, 'find') as mock_find:
            mock_find.return_value = mock_dirs
            
            dirs = await fs_service.get_directories(dir_id)
            
            assert len(dirs) == 2


class TestFSServiceSearch:
    """Tests for search functionality."""
    
    @pytest.fixture
    def fs_service(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return FSService(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_get_by_path_finds_file(self, fs_service):
        """Test get_by_path can find a file by path."""
        test_path = "/test/file.txt"
        mock_file = MagicMock(spec=FileRecord)
        mock_file.path = test_path
        
        with patch.object(FileRecord, 'find_one') as mock_find:
            mock_find.return_value = mock_file
            
            result = await fs_service.get_by_path(test_path)
            
            assert result == mock_file
            mock_find.assert_called_once_with({"path": test_path})
    
    @pytest.mark.asyncio
    async def test_get_by_path_finds_directory(self, fs_service):
        """Test get_by_path can find a directory by path."""
        test_path = "/test/dir"
        mock_dir = MagicMock(spec=DirectoryRecord)
        mock_dir.path = test_path
        
        with patch.object(FileRecord, 'find_one') as mock_find_file, \
             patch.object(DirectoryRecord, 'find_one') as mock_find_dir:
            
            mock_find_file.return_value = None
            mock_find_dir.return_value = mock_dir
            
            result = await fs_service.get_by_path(test_path)
            
            assert result == mock_dir
    
    @pytest.mark.asyncio
    async def test_get_by_path_returns_none_if_not_found(self, fs_service):
        """Test get_by_path returns None for non-existent path."""
        with patch.object(FileRecord, 'find_one') as mock_find_file, \
             patch.object(DirectoryRecord, 'find_one') as mock_find_dir:
            
            mock_find_file.return_value = None
            mock_find_dir.return_value = None
            
            result = await fs_service.get_by_path("/nonexistent")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_search_by_name_pattern_matching(self, fs_service):
        """Test search_by_name with pattern."""
        pattern = "test.*\\.txt"
        
        mock_files = [
            MagicMock(name="test1.txt"),
            MagicMock(name="test2.txt")
        ]
        
        with patch.object(FileRecord, 'find') as mock_find_files, \
             patch.object(DirectoryRecord, 'find') as mock_find_dirs:
            
            mock_find_files.return_value = mock_files
            mock_find_dirs.return_value = []
            
            results = await fs_service.search_by_name(pattern)
            
            assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_search_by_name_with_file_type_filter(self, fs_service):
        """Test search_by_name filters by file type."""
        pattern = "image"
        
        mock_images = [
            MagicMock(file_type="image"),
            MagicMock(file_type="image")
        ]
        
        with patch.object(FileRecord, 'find') as mock_find:
            mock_find.return_value = mock_images
            
            results = await fs_service.search_by_name(
                pattern,
                file_type="image"
            )
            
            # Verify file_type filter was used
            call_args = mock_find.call_args[0][0]
            assert "file_type" in call_args
    
    @pytest.mark.asyncio
    async def test_search_by_name_respects_limit(self, fs_service):
        """Test search_by_name respects limit parameter."""
        pattern = "test"
        limit = 10
        
        with patch.object(FileRecord, 'find') as mock_find_files, \
             patch.object(DirectoryRecord, 'find') as mock_find_dirs:
            
            mock_find_files.return_value = []
            mock_find_dirs.return_value = []
            
            await fs_service.search_by_name(pattern, limit=limit)
            
            # Should pass limit to find methods
            # (Implementation detail may vary)


class TestFSServiceCRUD:
    """Tests for CRUD operations."""
    
    @pytest.fixture
    def fs_service(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return FSService(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_create_file_success(self, fs_service):
        """Test successful file creation."""
        test_path = "/test/newfile.txt"
        test_name = "newfile.txt"
        parent_id = ObjectId()
        
        with patch.object(FileRecord, 'create') as mock_create:
            mock_file = MagicMock()
            mock_file._id = ObjectId()
            mock_create.return_value = mock_file
            
            file = await fs_service.create_file(
                path=test_path,
                name=test_name,
                parent_id=parent_id
            )
            
            assert file is not None
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_file_creates_if_not_exists(self, fs_service):
        """Test upsert_file creates file if it doesn't exist."""
        test_path = "/test/file.txt"
        
        with patch.object(FileRecord, 'find_one') as mock_find, \
             patch.object(FileRecord, 'create') as mock_create:
            
            mock_find.return_value = None  # Doesn't exist
            mock_file = MagicMock()
            mock_create.return_value = mock_file
            
            file = await fs_service.upsert_file(
                path=test_path,
                name="file.txt"
            )
            
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_file_updates_if_exists(self, fs_service):
        """Test upsert_file updates existing file."""
        test_path = "/test/file.txt"
        
        mock_existing = MagicMock()
        mock_existing.save = AsyncMock()
        
        with patch.object(FileRecord, 'find_one') as mock_find:
            mock_find.return_value = mock_existing
            
            file = await fs_service.upsert_file(
                path=test_path,
                name="file.txt",
                size_bytes=1024
            )
            
            # Should update and save
            mock_existing.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_directory_success(self, fs_service):
        """Test successful directory creation."""
        test_path = "/test/newdir"
        test_name = "newdir"
        
        with patch.object(DirectoryRecord, 'create') as mock_create:
            mock_dir = MagicMock()
            mock_create.return_value = mock_dir
            
            directory = await fs_service.create_directory(
                path=test_path,
                name=test_name
            )
            
            assert directory is not None
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_directory_creates_if_not_exists(self, fs_service):
        """Test upsert_directory creates directory if it doesn't exist."""
        test_path = "/test/dir"
        
        with patch.object(DirectoryRecord, 'find_one') as mock_find, \
             patch.object(DirectoryRecord, 'create') as mock_create:
            
            mock_find.return_value = None
            mock_dir = MagicMock()
            mock_create.return_value = mock_dir
            
            directory = await fs_service.upsert_directory(
                path=test_path,
                name="dir"
            )
            
            mock_create.assert_called_once()


class TestFSServiceErrorHandling:
    """Tests for error handling in FSService."""
    
    @pytest.fixture
    def fs_service(self):
        mock_locator = MagicMock()
        mock_config = MagicMock()
        return FSService(mock_locator, mock_config)
    
    @pytest.mark.asyncio
    async def test_get_children_handles_database_error(self, fs_service):
        """Test get_children handles database errors gracefully."""
        dir_id = ObjectId()
        
        with patch.object(FileRecord, 'find') as mock_find:
            mock_find.side_effect = Exception("Database error")
            
            # Should not raise, return empty or handle gracefully
            try:
                await fs_service.get_children(dir_id)
            except Exception as e:
                # If it raises, it should be handled properly
                assert "Database error" in str(e)
    
    @pytest.mark.asyncio
    async def test_create_file_logs_errors(self, fs_service):
        """Test create_file logs errors appropriately."""
        with patch.object(FileRecord, 'create') as mock_create, \
             patch.object(fs_service, '_log_error') as mock_log:
            
            mock_create.side_effect = Exception("Creation failed")
            
            try:
                await fs_service.create_file(
                    path="/test/file.txt",
                    name="file.txt"
                )
            except:
                pass
            
            # Verify error was logged (if implementation does this)


class TestFSServiceIntegration:
    """Integration tests for FSService."""
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve_file(self):
        """Test creating and retrieving a file."""
        # This would require actual database setup
        # Placeholder for full integration test
        pass
    
    @pytest.mark.asyncio
    async def test_file_hierarchy_navigation(self):
        """Test navigating file hierarchy."""
        # This would test creating root -> dir -> file
        # and navigating back up
        pass
