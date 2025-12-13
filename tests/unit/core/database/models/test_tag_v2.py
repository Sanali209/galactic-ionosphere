import pytest
import asyncio
from bson import ObjectId
from src.core.database.models.tag import Tag, TagManager

@pytest.fixture(scope="function")
async def db_teardown_tags():
    """Specific cleanup for tags"""
    from src.core.database.manager import db_manager
    from src.core.locator import sl
    from unittest.mock import MagicMock
    
    # Mock config
    mock_cfg = MagicMock()
    mock_cfg.data.mongo.host = "localhost"
    mock_cfg.data.mongo.port = 27017
    mock_cfg.data.mongo.database_name = "ionosphere_test"
    sl.config = mock_cfg
    
    # Just init (overwriting old client is safer than trying to close it from wrong loop)
    db_manager.init()
        
    if db_manager.db is not None:
        await db_manager.db.drop_collection("tags")
        
    yield
    
    # Teardown: Close client while we are still on the correct loop
    if db_manager.db is not None:
        await db_manager.db.drop_collection("tags")
    
    if db_manager.client:
        await db_manager.client.close()
        db_manager.client = None
        db_manager.db = None

@pytest.mark.asyncio
async def test_tag_fullname_creation(db_teardown_tags):
    # Root Tag
    root = await TagManager.create_tag("Root")
    assert root.fullName == "Root"
    
    # Child Tag
    child = await TagManager.create_tag("Child", parent=root)
    assert child.fullName == "Root/Child"
    
    # Grandchild Tag
    gc = await TagManager.create_tag("GrandChild", parent=child)
    assert gc.fullName == "Root/Child/GrandChild"

@pytest.mark.asyncio
async def test_tag_move_updates_fullname(db_teardown_tags):
    # Setup: RootA -> Child, RootB
    root_a = await TagManager.create_tag("RootA")
    child = await TagManager.create_tag("Child", parent=root_a)
    assert child.fullName == "RootA/Child"
    
    root_b = await TagManager.create_tag("RootB")
    
    # Move Child to RootB
    await TagManager.move_tag(child, new_parent=root_b)
    
    # Reload
    child = await Tag.get(child.id)
    assert child.fullName == "RootB/Child"
    assert child.parent_id == root_b.id

@pytest.mark.asyncio
async def test_ensure_from_path(db_teardown_tags):
    # Ensure "Animals|Cats|Persian"
    leaf = await TagManager.ensure_from_path("Animals|Cats|Persian")
    assert leaf.name == "Persian"
    assert leaf.fullName == "Animals/Cats/Persian"
    
    # Verify parents exist
    cats = await Tag.find_one({"name": "Cats"})
    assert cats is not None
    assert cats.fullName == "Animals/Cats"
    
    animals = await Tag.find_one({"name": "Animals"})
    assert animals is not None
    assert animals.fullName == "Animals"
