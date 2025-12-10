import pytest
from bson import ObjectId
from src.core.database.models.reference import RelationManager, Reference

@pytest.fixture(scope="function")
async def db_teardown_refs():
    from src.core.database.manager import db_manager
    if db_manager.db is not None:
        await db_manager.db.drop_collection("references")
    yield
    if db_manager.db is not None:
        await db_manager.db.drop_collection("references")

@pytest.mark.asyncio
async def test_create_relation(db_teardown_refs):
    src = ObjectId()
    tgt = ObjectId()
    
    ref = await RelationManager.link(src, tgt, "SIMILAR", {"score": 0.9})
    
    assert ref.id is not None
    assert ref.source_id == src
    assert ref.target_id == tgt
    assert ref.rel_type == "SIMILAR"
    assert ref.payload["score"] == 0.9

@pytest.mark.asyncio
async def test_find_relations(db_teardown_refs):
    src = ObjectId()
    t1 = ObjectId()
    t2 = ObjectId()
    
    await RelationManager.link(src, t1, "SIMILAR")
    await RelationManager.link(src, t2, "DUPLICATE")
    
    # Get all outgoing
    refs = await RelationManager.get_related(src)
    assert len(refs) == 2
    
    # Filter by type
    sims = await RelationManager.get_related(src, "SIMILAR")
    assert len(sims) == 1
    assert sims[0].target_id == t1

@pytest.mark.asyncio
async def test_unlink(db_teardown_refs):
    src = ObjectId()
    tgt = ObjectId()
    
    await RelationManager.link(src, tgt, "LINK")
    assert len(await RelationManager.get_related(src)) == 1
    
    await RelationManager.unlink(src, tgt, "LINK")
    assert len(await RelationManager.get_related(src)) == 0
