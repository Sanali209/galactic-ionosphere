import pytest
from datetime import datetime
from bson import ObjectId
from src.core.database.orm import (
    CollectionRecord, StringField, ReferenceField, 
    ListField, EmbeddedField, Reference
)

# --- Mocks ---
class Profile(CollectionRecord):
    _collection_name = "profiles"
    bio = StringField()

class User(CollectionRecord):
    _collection_name = "users"
    name = StringField(index=True)
    profile = ReferenceField(Profile) # 1:1
    tags = ListField(StringField())   # List
    
class Group(CollectionRecord):
    _collection_name = "groups"
    members = ListField(ReferenceField(User)) # 1:N / M:N

# --- Tests ---

def test_fields_serialization():
    # Test Embedded/List
    u = User(name="Test", tags=["a", "b"])
    data = u.to_dict()
    assert data['name'] == "Test"
    assert data['tags'] == ["a", "b"]
    assert data['_cls'] == "User"

def test_reference_handling():
    # Mocking DB interactions usually requires async mock
    # Here we test structure logic
    
    p = Profile(bio="dev")
    u = User(name="Alice", profile=p)
    
    # Save behavior (to_dict)
    data = u.to_dict()
    assert isinstance(data['profile'], ObjectId)
    assert data['profile'] == p.id
    
    # Load behavior (from_mongo)
    # Simulate loading from DB
    raw_data = {
        "_id": ObjectId(),
        "_cls": "User",
        "name": "Bob",
        "profile": p.id,
        "tags": []
    }
    
    loaded_user = User._instantiate_from_data(raw_data)
    assert isinstance(loaded_user.profile, Reference)
    assert loaded_user.profile.id == p.id
    
    # We can't test fetch() without a running event loop and DB connection
    # but we verify the Reference object is correctly created.

def test_indexes_structure():
    assert User._fields['name'].index is True
    # ensure_indexes would iterate this
