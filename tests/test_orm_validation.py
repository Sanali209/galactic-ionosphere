import pytest
from src.core.database.orm import CollectionRecord, StringField, IntField, ReferenceList

class User(CollectionRecord):
    name = StringField()
    age = IntField()

def test_string_field_validation():
    user = User()
    user.name = "Alice"
    assert user.name == "Alice"
    
    # Should enforce string
    user.name = 123
    assert user.name == "123" # Current behavior casts to string

def test_int_field_validation():
    user = User()
    user.age = 30
    assert user.age == 30
    
    with pytest.raises(TypeError):
        user.age = "invalid"

def test_reference_list_add():
    ref_list = ReferenceList([])
    
    # Needs ref_cls for raw ID
    user = User()
    
    # Test adding record
    ref_list.add(user)
    assert len(ref_list) == 1
    
    # Test adding string (ObjectId)
    # We need a valid objectid string
    valid_id = "507f1f77bcf86cd799439011"
    ref_list.add(valid_id, ref_cls=User)
    assert len(ref_list) == 2
