import pytest
from bson import ObjectId
from src.core.database.models.base import BaseEntity
from src.core.database.models.image import ImageRecord

# --- Test Fixtures in conftest.py handle DB init ---

@pytest.mark.asyncio
async def test_image_creation(db_teardown):
    img = ImageRecord(path="/tmp", filename="test.jpg")
    img.content_md5 = "abc123hash"
    img.rating = 5
    
    await img.save()
    
    assert img.id is not None
    assert img.full_path.replace("\\", "/") == "/tmp/test.jpg"
    
    # Verify _cls in DB
    from src.core.database.manager import db_manager
    raw_doc = await db_manager.db["gallery_entities"].find_one({"_id": img.id})
    assert raw_doc["_cls"] == "ImageRecord"

@pytest.mark.asyncio
async def test_tag_assignment(db_teardown):
    img = ImageRecord(filename="tagged.jpg")
    tag_id = ObjectId()
    
    img.add_tag(tag_id)
    assert tag_id in img.tag_ids
    
    await img.save()
    
    loaded = await ImageRecord.get(img.id)
    assert tag_id in loaded.tag_ids
    
    # Remove
    loaded.remove_tag(tag_id)
    assert tag_id not in loaded.tag_ids
    await loaded.save()

@pytest.mark.asyncio
async def test_polymorphism_query(db_teardown):
    # Save an ImageRecord
    img = ImageRecord(filename="poly.jpg")
    await img.save()
    
    # Query via BaseEntity
    # Since ImageRecord inherits BaseEntity, BaseEntity.get() should work 
    # IF they share the table "gallery_entities".
    # ImageRecord(BaseEntity) -> BaseEntity(CollectionRecord, table="gallery_entities")
    # Yes, ImageRecord inherits the table.
    
    base_obj = await BaseEntity.get(img.id)
    assert isinstance(base_obj, ImageRecord)
    assert base_obj.filename == "poly.jpg"
