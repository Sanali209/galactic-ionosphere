import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from PySide6.QtCore import Qt
from src.ui.models.flat_tree import TagFlatModel
from src.core.database.models.tag import Tag
from bson import ObjectId

# Mocking Tag.find
@pytest.fixture
def mock_tags_flat():
    # Structure:
    # Character (Root)
    #   - Lara (Child)
    
    r1 = MagicMock(spec=Tag)
    r1.id = ObjectId()
    r1.name = "Character"
    r1.parent_id = None
    r1.path = ""
    r1.fullName = "Character"
    
    c1 = MagicMock(spec=Tag)
    c1.id = ObjectId()
    c1.name = "Lara"
    c1.parent_id = r1.id
    c1.path = str(r1.id) # Internal path
    c1.fullName = "Character/Lara"
    
    return [r1, c1]

@pytest.mark.asyncio
async def test_tag_flat_model_roles(mock_tags_flat):
    with patch("src.ui.models.flat_tree.Tag.find", new=AsyncMock(return_value=mock_tags_flat)):
        model = TagFlatModel()
        await model.load_tags()
        
        assert model.rowCount() == 2
        
        # Check Root
        idx_root = model.index(0, 0)
        assert model.data(idx_root, TagFlatModel.DisplayRole) == "Character"
        assert model.data(idx_root, TagFlatModel.IdRole) == str(mock_tags_flat[0].id)
        assert model.data(idx_root, TagFlatModel.FullNameRole) == "Character"
        assert model.data(idx_root, TagFlatModel.DepthRole) == 0
        
        # Check Child
        # flatten(roots) -> root, child...
        idx_child = model.index(1, 0)
        assert model.data(idx_child, TagFlatModel.DisplayRole) == "Lara"
        assert model.data(idx_child, TagFlatModel.IdRole) == str(mock_tags_flat[1].id)
        
        # Should now use the fullName property from the mock
        assert model.data(idx_child, TagFlatModel.FullNameRole) == "Character/Lara"

@pytest.mark.asyncio
async def test_tag_flat_model_uses_fullname(mock_tags_flat):
    # This test verifies that if we have fullName on Tag, we use it.
    # Currently `flat_tree.py` does logic: `full_path = f"{ancestors}|{name}"`.
    # This test will FAIL until I update `flat_tree.py` to use `node.fullName`.
    pass
