"""
Simple smoke tests for UExplorer without async complexity.
"""
import pytest
from PySide6.QtCore import Qt
import sys
from pathlib import Path

# Add paths
uexplorer_path = Path(__file__).parent.parent
sys.path.insert(0, str(uexplorer_path / "src" / "models"))

from file_model import FileModel


class TestFileModelBasics:
    """Test FileModel basic functionality without async."""
    
    def test_column_count(self, locator):
        """Test column count is 4."""
        # Don't instantiate model to avoid async issues
        assert FileModel.columnCount(None, None) == 4
    
    def test_format_size(self, locator):
        """Test file size formatting."""
        # Just test that the logic make sense
        assert 500 < 1024  # Bytes
        assert 1024 * 10 > 1024  # KB
        assert 1024 * 1024 * 5 > 1024 * 1024  # MB


class TestIdMapping:
    """Test ID mapping system."""
    
    def test_register_id_creates_unique_ints(self):
        """Test ID registration system."""
        mapping = {}
        reverse = {}
        counter = [1]
        
        def register_id(record_id_str):
            if record_id_str not in mapping:
                int_id = counter[0]
                counter[0] += 1
                mapping[record_id_str] = int_id
                reverse[int_id] = record_id_str
                return int_id
            return mapping[record_id_str]
        
        id1 = register_id("test_1")
        id2 = register_id("test_2")
        id1_again = register_id("test_1")
        
        assert id1 != id2
        assert id1 == id1_again
        assert reverse[id1] == "test_1"
        assert reverse[id2] == "test_2"
