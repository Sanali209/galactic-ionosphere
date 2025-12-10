import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from PySide6.QtCore import Qt
from src.ui.models.navigation import TagTreeModel
from src.core.database.models.tag import Tag
from bson import ObjectId

# Mocking Tag.find
@pytest.fixture
def mock_tags():
    # Structure:
    # Root1
    #   - Child1
    # Root2
    
    r1 = MagicMock(spec=Tag)
    r1.id = ObjectId()
    r1.name = "Root1"
    r1.parent_id = None
    r1.path = ""
    
    c1 = MagicMock(spec=Tag)
    c1.id = ObjectId()
    c1.name = "Child1"
    c1.parent_id = r1.id
    c1.path = str(r1.id)
    
    r2 = MagicMock(spec=Tag)
    r2.id = ObjectId()
    r2.name = "Root2"
    r2.parent_id = None
    r2.path = ""
    
    return [r1, c1, r2]

@pytest.mark.asyncio
async def test_tag_tree_model_structure(mock_tags):
    with patch("src.ui.models.navigation.Tag.find", new=AsyncMock(return_value=mock_tags)):
        model = TagTreeModel()
        await model._fetch_data() # Manual trigger of async load
        
        # Test Roots
        assert model.rowCount() == 2
        
        # Test Root1 Data
        idx_r1 = model.index(0, 0)
        assert idx_r1.isValid()
        assert model.data(idx_r1, Qt.DisplayRole) == "Root1"
        
        # Test Child1 (under Root1)
        assert model.rowCount(idx_r1) == 1
        idx_c1 = model.index(0, 0, idx_r1)
        assert idx_c1.isValid()
        assert model.data(idx_c1, Qt.DisplayRole) == "Child1"
        assert model.data(idx_c1, TagTreeModel.IdRole) == str(mock_tags[1].id)
        
        # Test Root2 Data
        idx_r2 = model.index(1, 0)
        assert idx_r2.isValid()
        assert model.data(idx_r2, Qt.DisplayRole) == "Root2"
        assert model.rowCount(idx_r2) == 0
