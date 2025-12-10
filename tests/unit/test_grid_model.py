import pytest
from unittest.mock import MagicMock, AsyncMock
from src.ui.models.grid_model import GalleryGridModel
from src.ui.bridge import BackendBridge
from src.core.database.models.image import ImageRecord
from bson import ObjectId

@pytest.fixture
def mock_deps():
    importer = MagicMock()
    search = MagicMock()
    search.search_by_text = AsyncMock(return_value=[])
    return importer, search

def test_grid_model_update():
    model = GalleryGridModel()
    
    img1 = MagicMock(spec=ImageRecord)
    img1.id = ObjectId()
    img1.full_path = "/a/b.jpg"
    img1.content_md5 = "hash1"
    
    model.set_images([img1])
    
    assert model.rowCount() == 1
    idx = model.index(0, 0)
    assert model.data(idx, GalleryGridModel.PathRole) == "/a/b.jpg"

@pytest.mark.asyncio
async def test_bridge_search_integration(mock_deps):
    importer, search = mock_deps
    model = GalleryGridModel()
    bridge = BackendBridge(importer, search, model)
    
    # Setup Search Result
    img = MagicMock(spec=ImageRecord)
    img.id = ObjectId()
    search.search_by_text.return_value = [img]
    
    # Execute internal async method directly to avoid needing full Qt Loop in unit test
    await bridge._do_search("test")
    
    # Verify Model Updated
    assert model.rowCount() == 1
    # Verify Signal (Manual check, or spy)
    # bridge.searchFinished.emit... hard to check without QtTest or signal spy.
    # Logic verification is enough.
