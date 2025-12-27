import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from mongomock import MongoClient
from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.mongoODM.client_component import MongoClientComponent
from SLM.core.mongoODM.database_manager import DatabaseManagerComponent
from SLM.core.mongoODM.documents import Document
from SLM.core.mongoODM.fields import StringField, IntField, GenericReferenceField

@pytest.fixture
def mock_config():
    """Fixture to create a mock Config object."""
    conf = Config()
    conf.set_value("mongodb", {
        "host": "localhost", 
        "port": 27017, 
        "db_name": "test_db"
    })
    return conf

# Define a simple Document for testing
class User(Document):
    __collection__ = "users"
    name = StringField(required=True)
    age = IntField()

@pytest.fixture
def mock_mongo_client():
    """Fixture to create a mock MongoClient."""
    return MongoClient()

@pytest.fixture
def odm_components(mock_mongo_client, mock_config):
    """Fixture to set up the ODM components for testing."""
    # Mock the client component to use mongomock
    client_comp = MongoClientComponent(config=mock_config)
    client_comp.client = mock_mongo_client
    
    # Create the message bus and database manager
    message_bus = MessageBus()
    db_manager = DatabaseManagerComponent(client_component=client_comp, message_bus=message_bus)
    
    # Manually start the components
    db_manager.start()
    
    return client_comp, db_manager, message_bus

def test_create_user(odm_components):
    """Test creating a new user document."""
    _, _, message_bus = odm_components
    
    events = []
    def on_user_created(msg_type, document):
        events.append(document)
    
    message_bus.subscribe("document.user.created", on_user_created)

    user = User(name="John Doe", age=30)
    user.save()

    assert user.pk is not None
    assert user.name == "John Doe"
    assert len(events) == 1
    assert events[0].pk == user.pk

def test_find_user(odm_components):
    """Test finding a user document."""
    # Create a user first
    user = User(name="Jane Doe", age=25)
    user.save()

    # Now find it
    found_user = User.objects.find_one({"name": "Jane Doe"})
    
    assert found_user is not None
    assert found_user.pk == user.pk
    assert found_user.age == 25

def test_update_user(odm_components):
    """Test updating a user document."""
    _, _, message_bus = odm_components
    
    events = []
    def on_user_updated(msg_type, document):
        events.append(document)
    
    message_bus.subscribe("document.user.updated", on_user_updated)

    user = User(name="Test User", age=40)
    user.save()
    
    user.age = 41
    user.save()

    updated_user = User.objects.find_one({"_id": user.pk})
    assert updated_user.age == 41
    assert len(events) == 1
    assert events[0].age == 41

def test_delete_user(odm_components):
    """Test deleting a user document."""
    _, _, message_bus = odm_components
    
    events = []
    def on_user_deleted(msg_type, document_id):
        events.append(document_id)
    
    message_bus.subscribe("document.user.deleted", on_user_deleted)

    user = User(name="Delete Me", age=99)
    user.save()
    user_id = user.pk

    user.delete()

    deleted_user = User.objects.find_one({"_id": user_id})
    assert deleted_user is None
    assert user.pk is None
    assert len(events) == 1
    assert events[0] == user_id

# --- Tests for New Features ---

# 1. Test for Compound Indexes
class Product(Document):
    __collection__ = "products"
    name = StringField()
    category = StringField()

    class Meta:
        indexes = [
            [('name', 1), ('category', -1)]
        ]

def test_compound_index_creation(odm_components):
    """Test that compound indexes defined in Meta are correctly registered."""
    # This test verifies that the metaclass correctly collects the index info.
    # The actual index creation is mocked but this confirms the definition is processed.
    assert len(Product._indexes) > 0
    
    # Check for the specific compound index
    compound_index_found = False
    for index in Product._indexes:
        if isinstance(index, list) and index == [('name', 1), ('category', -1)]:
            compound_index_found = True
            break
    assert compound_index_found

# 2. Test for GenericReferenceField
class Book(Document):
    __collection__ = "books"
    title = StringField()

class Movie(Document):
    __collection__ = "movies"
    title = StringField()

class Review(Document):
    __collection__ = "reviews"
    content = StringField()
    item = GenericReferenceField()

def test_generic_reference_field(odm_components):
    """Test creating and resolving a GenericReferenceField."""
    # Create items to be reviewed
    book = Book(title="The Hitchhiker's Guide to the Galaxy")
    book.save()

    movie = Movie(title="The Matrix")
    movie.save()

    # Create reviews for both items
    review_for_book = Review(content="A fantastic read!", item=book)
    review_for_book.save()

    review_for_movie = Review(content="A mind-bending classic!", item=movie)
    review_for_movie.save()

    # Add assertions to ensure the objects managers are not None
    assert Review.objects is not None
    assert Book.objects is not None
    assert Movie.objects is not None

    # Fetch the reviews and check the references
    fetched_review_book = Review.objects.find_one({"_id": review_for_book.pk})
    fetched_review_movie = Review.objects.find_one({"_id": review_for_movie.pk})

    # Assert that the fetched documents are not None
    assert fetched_review_book is not None
    assert fetched_review_movie is not None

    # Assert that the references are resolved to the correct document instances
    assert isinstance(fetched_review_book.item, Book)
    assert fetched_review_book.item.pk == book.pk
    assert fetched_review_book.item.title == "The Hitchhiker's Guide to the Galaxy"

    assert isinstance(fetched_review_movie.item, Movie)
    assert fetched_review_movie.item.pk == movie.pk
    assert fetched_review_movie.item.title == "The Matrix"
