"""
UCoreFS - File Operations Tests

Tests for FSService file move, copy, and rename operations.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId


class TestFSServiceFileOperations:
    """Tests for FSService file operations."""
    
    @pytest.fixture
    def sample_file(self):
        """Create a sample test file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_file.txt")
            with open(file_path, "w") as f:
                f.write("Test content")
            yield file_path, tmpdir
    
    def test_fs_service_has_move_file_method(self):
        """Test FSService has move_file method."""
        from src.core.services.fs_service import FSService
        
        assert hasattr(FSService, 'move_file')
    
    def test_fs_service_has_copy_file_method(self):
        """Test FSService has copy_file method."""
        from src.core.services.fs_service import FSService
        
        assert hasattr(FSService, 'copy_file')
    
    def test_fs_service_has_rename_file_method(self):
        """Test FSService has rename_file method."""
        from src.core.services.fs_service import FSService
        
        assert hasattr(FSService, 'rename_file')
    
    def test_get_unique_path(self):
        """Test _get_unique_path generates unique filenames."""
        from src.core.services.fs_service import FSService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file
            existing = Path(tmpdir) / "file.txt"
            existing.touch()
            
            # Get unique path
            unique = service._get_unique_path(existing)
            
            assert unique != existing
            assert "file_1.txt" in str(unique)
    
    @pytest.mark.asyncio
    async def test_move_file_not_found(self):
        """Test move_file returns error for non-existent file."""
        from src.core.services.fs_service import FSService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        # Mock FileRecord.get to return None
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_get.return_value = None
            
            result = await service.move_file(ObjectId(), "/some/path")
            
            assert result["success"] == False
            assert "not found" in result.get("error", "").lower()
    
    @pytest.mark.asyncio
    async def test_copy_file_not_found(self):
        """Test copy_file returns error for non-existent file."""
        from src.core.services.fs_service import FSService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        # Mock FileRecord.get to return None
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_get.return_value = None
            
            result = await service.copy_file(ObjectId(), "/some/path")
            
            assert result["success"] == False
            assert "not found" in result.get("error", "").lower()
    
    @pytest.mark.asyncio
    async def test_rename_file_not_found(self):
        """Test rename_file returns error for non-existent file."""
        from src.core.services.fs_service import FSService
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        # Mock FileRecord.get to return None
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_get.return_value = None
            
            result = await service.rename_file(ObjectId(), "new_name.txt")
            
            assert result["success"] == False
            assert "not found" in result.get("error", "").lower()
    
    @pytest.mark.asyncio
    async def test_move_file_success(self, sample_file):
        """Test successful file move."""
        from src.core.services.fs_service import FSService
        
        file_path, tmpdir = sample_file
        dest_folder = os.path.join(tmpdir, "destination")
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        # Mock FileRecord
        mock_file = MagicMock()
        mock_file.path = file_path
        mock_file.name = "test_file.txt"
        mock_file.save = AsyncMock()
        
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_get.return_value = mock_file
            
            result = await service.move_file(ObjectId(), dest_folder)
            
            assert result["success"] == True
            assert "new_path" in result
            assert dest_folder in result["new_path"]
    
    @pytest.mark.asyncio
    async def test_rename_file_success(self, sample_file):
        """Test successful file rename."""
        from src.core.services.fs_service import FSService
        
        file_path, tmpdir = sample_file
        
        mock_locator = MagicMock()
        mock_config = MagicMock()
        
        service = FSService(mock_locator, mock_config)
        
        # Mock FileRecord
        mock_file = MagicMock()
        mock_file.path = file_path
        mock_file.name = "test_file.txt"
        mock_file.save = AsyncMock()
        
        with patch('src.ucorefs.models.file_record.FileRecord.get') as mock_get:
            mock_get.return_value = mock_file
            
            result = await service.rename_file(ObjectId(), "renamed_file.txt")
            
            assert result["success"] == True
            assert result["old_name"] == "test_file.txt"
            assert "renamed_file.txt" in result["new_path"]
    
    def test_conflict_resolution_options(self):
        """Test conflict resolution options are documented."""
        from src.core.services.fs_service import FSService
        
        # Check docstring mentions all options
        docstring = FSService.move_file.__doc__
        
        assert "rename" in docstring
        assert "skip" in docstring
        assert "overwrite" in docstring
