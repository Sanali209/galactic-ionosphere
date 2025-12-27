import sys
import os
import asyncio
from pymongo import MongoClient

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.mongoODM.db_component import MongoODMComponent
from SLM.core.mongoODM.documents import Document, EmbeddedDocument
from SLM.core.mongoODM.fields import StringField, IntField, EmbeddedDocumentField, ListField, ReferenceField, ReverseReferenceField

# --- 1. Define Document Schemas ---

# Example of an Embedded Document
class Address(EmbeddedDocument):
    street = StringField(required=True)
    city = StringField(required=True)

# Example of a main Document with an embedded one
class User(Document):
    __collection__ = "users"
    name = StringField(required=True, index=True)
    email = StringField(required=True, unique=True)
    address = EmbeddedDocumentField(Address)

# Example of Document Inheritance
class Media(Document):
    __collection__ = "media"
    __abstract__ = True  # This document won't be instantiated directly
    title = StringField(required=True)

class Author(Document):
    __collection__ = "authors"
    name = StringField(required=True)
    books = ReverseReferenceField('Book', 'author')

class Book(Media):
    author = ReferenceField(Author, required=True)
    pages = IntField()

class Movie(Media):
    director = StringField(required=True)
    duration_mins = IntField()

# Example of a Document with References
class Library(Document):
    __collection__ = "libraries"
    name = StringField(required=True)
    items = ListField(ReferenceField(Media))


# --- 2. Setup Application Components ---

def setup_components():
    """Sets up and connects the necessary components for the ODM to work."""
    # Clean the database for a fresh run
    client = MongoClient('localhost', 27017)
    client.drop_database('slm_odm_example_db')
    client.close()

    # a. Configuration
    config = Config()
    config.set_value("mongodb", {
        "host": "localhost",
        "port": 27017,
        "db_name": "slm_odm_example_db"
    })

    # b. Message Bus
    message_bus = MessageBus()

    # c. Unified ODM Component
    odm_component = MongoODMComponent(config=config, message_bus=message_bus)

    # d. Start component
    odm_component.start()

    print("--- Components Setup Complete ---")
    return odm_component

# --- 3. Run Examples ---

def run_examples():
    """Execute demonstrations of the ODM features."""
    print("\n--- Running ODM Examples ---")

    # --- Embedded Document Example ---
    print("\n1. Testing Embedded Documents...")
    user_address = Address(street="123 Python Lane", city="Codeville")
    user = User(name="John Doe", email="john.doe@example.com", address=user_address)
    user.save()
    print(f"Saved User: {user.name} with address: {user.address.city}")

    found_user = User.objects.find_one({"email": "john.doe@example.com"})
    print(f"Found User: {found_user.name}, Street: {found_user.address.street}")
    assert found_user.address.street == "123 Python Lane"
    print("   Embedded Document Test PASSED.")

    # --- Inheritance Example ---
    print("\n2. Testing Document Inheritance...")
    author_tim = Author(name="Tim Peters").save()
    book = Book(title="The Zen of Python", author=author_tim, pages=1)
    movie = Movie(title="The Social Network", director="David Fincher", duration_mins=120)
    book.save()
    movie.save()
    print(f"Saved Book: {book.title}")
    print(f"Saved Movie: {movie.title}")

    # Find all media items - should get both Book and Movie
    all_media = list(Media.objects.find())
    print(f"Found {len(all_media)} media items in the collection.")
    assert len(all_media) == 2
    
    # Polymorphic loading: check that we get back the correct types
    found_book = Media.objects.find_one({"title": "The Zen of Python"})
    found_movie = Media.objects.find_one({"title": "The Social Network"})
    
    assert found_book.author.name == "Tim Peters"
    
    assert isinstance(found_book, Book)
    assert isinstance(found_movie, Movie)
    print(f"   Polymorphically loaded a {type(found_book).__name__} and a {type(found_movie).__name__}.")
    print("   Document Inheritance Test PASSED.")

    # --- Reference Field Example ---
    print("\n3. Testing Reference Fields...")
    library = Library(name="The Grand Archives")
    library.items = [book, movie] # Assign the actual objects
    library.save()
    print(f"Saved Library: '{library.name}' with {len(library.items)} items.")

    # Find the library and check its references
    found_library = Library.objects.find_one({"name": "The Grand Archives"})
    
    # The 'items' are stored as DBRefs but should be loaded as objects
    # Note: Automatic dereferencing is not implemented in this simple ODM,
    # so we'll check the stored references. A full-featured ODM would auto-fetch.
    
    # Let's manually fetch and verify
    # The 'items' are loaded as raw ObjectIds, so we can use them directly.
    item_ids = found_library.items
    referenced_items = list(Media.objects.find({"_id": {"$in": item_ids}}))
    
    print(f"   Found {len(referenced_items)} items by dereferencing.")
    assert len(referenced_items) == 2
    titles = {item.title for item in referenced_items}
    assert "The Zen of Python" in titles
    assert "The Social Network" in titles
    print("   Reference Field Test PASSED.")

    # --- Reverse Reference Field Example ---
    print("\n4. Testing Reverse Reference Fields...")
    author_dahl = Author(name="Roald Dahl").save()
    book1 = Book(title="Charlie and the Chocolate Factory", author=author_dahl, pages=192).save()
    book2 = Book(title="Matilda", author=author_dahl, pages=240).save()

    # Find the author and check their books
    found_author = Author.objects.find_one({"name": "Roald Dahl"})
    author_books = list(found_author.books)
    
    print(f"   Found {len(author_books)} books for author '{found_author.name}'.")
    assert len(author_books) == 2
    
    book_titles = {b.title for b in author_books}
    assert "Matilda" in book_titles
    assert "Charlie and the Chocolate Factory" in book_titles
    print("   Reverse Reference Field Test PASSED.")

    print("\n--- All Examples Completed Successfully! ---")


if __name__ == "__main__":
    setup_components()
    run_examples()
