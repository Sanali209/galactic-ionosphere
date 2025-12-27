import pytest
from unittest.mock import AsyncMock, patch
from bson import ObjectId
from src.core.database.orm import (
    CollectionRecord, StringField, ReferenceField, 
    ListField, Reference, ReferenceList
)

# --- Mocks ---
class Tag(CollectionRecord):
    _collection_name = "tags"
    name = StringField()

class Post(CollectionRecord):
    _collection_name = "posts"
    title = StringField()
    tags = ListField(ReferenceField(Tag))

# --- Tests ---
def test_reference_list_serialization():
    t1 = Tag(name="Python")
    t2 = Tag(name="AI")
    
    # Create Post with tags
    post = Post(title="My Post", tags=[t1, t2])
    
    # 1. To Dict (Serialization) -> Should be list of IDs
    data = post.to_dict()
    assert len(data['tags']) == 2
    assert isinstance(data['tags'][0], ObjectId)
    assert data['tags'][0] == t1.id

def test_reference_list_deserialization():
    t1_id = ObjectId()
    t2_id = ObjectId()
    
    raw_data = {
        "_id": ObjectId(),
        "_cls": "Post",
        "title": "My Post",
        "tags": [t1_id, t2_id]
    }
    
    # 2. From Dict (Deserialization) -> Should be ReferenceList
    post = Post._instantiate_from_data(raw_data)
    
    assert isinstance(post.tags, ReferenceList)
    assert len(post.tags) == 2
    assert isinstance(post.tags[0], Reference)
    assert post.tags[0].id == t1_id

@pytest.mark.asyncio
async def test_reference_list_fetch_all():
    t1_id = ObjectId()
    t2_id = ObjectId()
    
    # Provide simple wrapper for testing
    class MockRefList(ReferenceList):
        pass

    ref_list = MockRefList([Reference(Tag, t1_id), Reference(Tag, t2_id)])
    
    # Mock Tag.get to return dummy objects
    with patch.object(Tag, 'get', side_effect=[Tag(oid=t1_id, name="Python"), Tag(oid=t2_id, name="AI")]) as mock_get:
        items = await ref_list.fetch_all()
        
        assert len(items) == 2
        assert items[0].name == "Python"
        assert items[1].name == "AI"
        assert mock_get.call_count == 2
