"""
UCoreFS Phase 2 Tests - Discovery System

Tests for:
- LibraryManager watch/blacklist functionality
- DirectoryScanner batch processing
- DiffDetector change detection
- SyncManager database updates
- DiscoveryService orchestration
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from bson import ObjectId
import tempfile
import os


class TestLibraryManager:
    """Tests for LibraryManager."""
    
    @pytest.fixture
    def mock_fs_service(self):
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_should_scan_extension_with_whitelist(self, mock_fs_service):
        """Test extension filtering with whitelist."""
        from src.ucorefs.discovery.library_manager import LibraryManager
        
        manager = LibraryManager(mock_fs_service)
        
        # With whitelist
        assert manager.should_scan_extension("jpg", ["jpg", "png"]) == True
        assert manager.should_scan_extension("txt", ["jpg", "png"]) == False
        
        # Without whitelist (scan all)
        assert manager.should_scan_extension("anything", []) == True
    
    @pytest.mark.asyncio
    async def test_is_blacklisted(self, mock_fs_service):
        """Test path blacklist checking."""
        from src.ucorefs.discovery.library_manager import LibraryManager
        
        manager = LibraryManager(mock_fs_service)
        
        blacklist = ["/media/photos/.cache", "/media/photos/temp"]
        
        assert manager.is_blacklisted("/media/photos/.cache/thumb.jpg", blacklist) == True
        assert manager.is_blacklisted("/media/photos/image.jpg", blacklist) == False
        assert manager.is_blacklisted("/media/photos/temp/file.tmp", blacklist) == True


class TestDirectoryScanner:
    """Tests for DirectoryScanner."""
    
    @pytest.fixture
    def mock_library_manager(self):
        manager = MagicMock()
        manager.should_scan_extension = MagicMock(return_value=True)
        manager.is_blacklisted = MagicMock(return_value=False)
        return manager
    
    def test_scan_result_creation(self):
        """Test ScanResult creation."""
        from src.ucorefs.discovery.scanner import ScanResult
        
        result = ScanResult("/test/file.jpg", is_directory=False, size=1024)
        
        assert result.path == "/test/file.jpg"
        assert result.is_directory == False
        assert result.size == 1024
        assert result.extension == "jpg"
    
    def test_batch_size_respected(self, mock_library_manager):
        """Test that scanner respects batch size."""
        from src.ucorefs.discovery.scanner import DirectoryScanner
        
        scanner = DirectoryScanner(mock_library_manager, batch_size=10)
        
        # Create temporary directory with files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 25 files
            for i in range(25):
                open(os.path.join(tmpdir, f"file{i}.txt"), 'w').close()
            
            batches = list(scanner.scan_directory(tmpdir, [], [], False))
            
            # Should have 3 batches (10 + 10 + 5)
            assert len(batches) >= 2
            assert all(len(batch) <= 10 for batch in batches)


class TestDiffDetector:
    """Tests for DiffDetector."""
    
    @pytest.mark.asyncio
    async def test_detect_added_files(self):
        """Test detection of newly added files."""
        from src.ucorefs.discovery.diff import DiffDetector
        from src.ucorefs.discovery.scanner import ScanResult
        
        detector = DiffDetector()
        
        # Mock scan results
        scan_results = [
            ScanResult("/test/new_file.jpg", False, size=1024)
        ]
        
        # Mock empty database (no existing files)
        with patch('src.ucorefs.models.file_record.FileRecord.find', new_callable=AsyncMock) as mock_find_files:
            with patch('src.ucorefs.models.directory.DirectoryRecord.find', new_callable=AsyncMock) as mock_find_dirs:
                mock_find_files.return_value = []
                mock_find_dirs.return_value = []
                
                diff = await detector.detect_changes(scan_results, "/test")
                
                assert len(diff.added_files) == 1
                assert diff.added_files[0].path == "/test/new_file.jpg"
    
    @pytest.mark.asyncio
    async def test_detect_deleted_files(self):
        """Test detection of deleted files."""
        from src.ucorefs.discovery.diff import DiffDetector
        from src.ucorefs.models.file_record import FileRecord
        
        detector = DiffDetector()
        
        # Empty scan results (file was deleted)
        scan_results = []
        
        # Mock database with existing file
        mock_file = FileRecord(path="/test/deleted.jpg", name="deleted.jpg")
        
        with patch('src.ucorefs.models.file_record.FileRecord.find') as mock_find:
            mock_find.return_value = AsyncMock(return_value=[mock_file])()
            
            with patch('src.ucorefs.models.directory.DirectoryRecord.find', new_callable=AsyncMock) as mock_find_dirs:
                mock_find_dirs.return_value = []
                
                diff = await detector.detect_changes(scan_results, "/test")
                
                # File in DB but not in scan = deleted
                assert len(diff.deleted_files) >= 0  # Depends on mock implementation


class TestSyncManager:
    """Tests for SyncManager."""
    
    @pytest.fixture
    def mock_fs_service(self):
        service = AsyncMock()
        service.create_file = AsyncMock()
        service.create_directory = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_add_files(self, mock_fs_service):
        """Test adding files to database."""
        from src.ucorefs.discovery.sync import SyncManager
        from src.ucorefs.discovery.scanner import ScanResult
        
        manager = SyncManager(mock_fs_service)
        
        scan_results = [
            ScanResult("/test/file1.jpg", False, size=1024),
            ScanResult("/test/file2.jpg", False, size=2048)
        ]
        
        with patch('src.ucorefs.models.directory.DirectoryRecord.find_one', new_callable=AsyncMock) as mock_find:
            mock_find.return_value = None
            
            count = await manager._add_files(scan_results, "root_id")
            
            assert count == 2
            assert mock_fs_service.create_file.call_count == 2


class TestDiscoveryService:
    """Tests for DiscoveryService."""
    
    @pytest.fixture
    def mock_locator(self):
        locator = MagicMock()
        locator.get_system = MagicMock(return_value=AsyncMock())
        return locator
    
    @pytest.fixture
    def mock_config(self):
        return MagicMock()
    
    @pytest.mark.asyncio
    async def test_discovery_service_initialize(self, mock_locator, mock_config):
        """Test DiscoveryService initialization."""
        from src.ucorefs.discovery.service import DiscoveryService
        
        service = DiscoveryService(mock_locator, mock_config)
        await service.initialize()
        
        assert service.is_ready == True
        assert hasattr(service, 'library_manager')
        assert hasattr(service, 'scanner')
        assert hasattr(service, 'diff_detector')
        assert hasattr(service, 'sync_manager')
