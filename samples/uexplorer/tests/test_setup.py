"""
Tests for UExplorer configuration and setup.
"""
import pytest
from pathlib import Path


class TestConfiguration:
    """Test configuration setup."""
    
    def test_config_file_exists(self):
        """Test config.toml exists."""
        config_path = Path(__file__).parent.parent / "config.toml"
        assert config_path.exists()
    
    def test_src_directory_exists(self):
        """Test src directory structure exists."""
        src_path = Path(__file__).parent.parent / "src"
        assert src_path.exists()
        assert (src_path / "ui").exists()
        assert (src_path / "models").exists()
    
    def test_locator_initialization(self, locator):
        """Test ServiceLocator is initialized."""
        assert locator is not None
        assert hasattr(locator, 'config')
        assert hasattr(locator, 'get_system')


class TestSystemsAvailability:
    """Test all required systems are available."""
    
    def test_fs_service_available(self, locator):
        """Test FSService is registered."""
        from src.core.services.fs_service import FSService
        fs_service = locator.get_system(FSService)
        assert fs_service is not None
    
    def test_tag_manager_available(self, locator):
        """Test TagManager is registered."""
        from src.ucorefs.tags.manager import TagManager
        tag_mgr = locator.get_system(TagManager)
        assert tag_mgr is not None
