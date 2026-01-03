"""
Comprehensive tests for the Custom ORM (src.core.database.orm).
"""
import pytest
from unittest.mock import patch, MagicMock
from bson import ObjectId
from src.core.database.orm import (
    CollectionRecord, StringField, IntField, ReferenceField, 
    ListField, Reference, ReferenceList
)

# --- Test Models ---

class Profile(CollectionRecord):
    _collection_name = "profiles"
    bio = StringField()

class Tag(CollectionRecord):
    _collection_name = "tags"
    name = StringField()

class User(CollectionRecord):
    _collection_name = "users"
    name = StringField(index=True)
    age = IntField()
    profile = ReferenceField(Profile) # 1:1
    tags = ListField(ReferenceField(Tag)) # 1:N

# --- Tests ---

def test_field_validation():
    user = User()
    
    # StringField
    user.name = "Alice"
    assert user.name == "Alice"
    user.name = 123
    assert user.name == "123" # Auto-cast
    
    # IntField
    user.age = 30
    assert user.age == 30
    user.age = "30"
    assert user.age == 30 # Auto-cast
    
    with pytest.raises(TypeError):
        user.age = "invalid"

def test_serialization():
    # Setup
    p = Profile(_id=ObjectId(), bio="Developer")
    t1 = Tag(_id=ObjectId(), name="Python")
    t2 = Tag(_id=ObjectId(), name="AI")
    
    user = User(
        name="Bob",
        age=25,
        profile=p,
        tags=[t1, t2]
    )
    
    # Serialize
    data = user.to_dict()
    
    assert data["name"] == "Bob"
    assert data["age"] == 25
    assert data["profile"] == p.id
    assert len(data["tags"]) == 2
    assert data["tags"][0] == t1.id

def test_deserialization():
    p_id = ObjectId()
    t1_id = ObjectId()
    
    raw_data = {
        "_id": ObjectId(),
        "_cls": "User",
        "name": "Charlie",
        "age": 40,
        "profile": p_id,
        "tags": [t1_id]
    }
    
    user = User._instantiate_from_data(raw_data)
    
    assert user.name == "Charlie"
    assert user.age == 40
    
    # Check References
    assert isinstance(user.profile, Reference)
    assert user.profile.id == p_id
    assert user.profile.ref_cls == Profile
    
    assert isinstance(user.tags, ReferenceList)
    assert len(user.tags) == 1
    assert user.tags[0].id == t1_id

@pytest.mark.asyncio
async def test_reference_fetch():
    # Mocking fetch
    p_id = ObjectId()
    user = User(profile=Reference(Profile, p_id))
    
    # Mock Profile.get to return a Profile object
    mock_profile = Profile(oid=p_id, bio="Mocked Bio")
    
    with patch.object(Profile, 'get', return_value=mock_profile) as mock_get:
        fetched = await user.profile.fetch()
        
        assert fetched.bio == "Mocked Bio"
        mock_get.assert_called_once_with(p_id)

@pytest.mark.asyncio
async def test_reference_list_fetch_all():
    t1_id = ObjectId()
    t2_id = ObjectId()
    
    # Create ReferenceList manually
    ref_list = ReferenceList([
        Reference(Tag, t1_id),
        Reference(Tag, t2_id)
    ])
    
    mock_tags = [
        Tag(oid=t1_id, name="Tag1"),
        Tag(oid=t2_id, name="Tag2")
    ]
    
    with patch.object(Tag, 'get', side_effect=mock_tags) as mock_get:
        items = await ref_list.fetch_all()
        
        assert len(items) == 2
        assert items[0].name == "Tag1"
        assert items[1].name == "Tag2"
        assert mock_get.call_count == 2
