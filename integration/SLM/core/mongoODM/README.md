# SLM Core - MongoDB ODM Framework

This directory contains a lightweight, component-based Object-Document Mapper (ODM) for MongoDB, designed to integrate seamlessly with the SLM core framework.

## Features

- **Automatic Background Persistence**: All field changes save immediately to the database without explicit save() calls
- **Declarative Schemas**: Define your document structure using simple Python classes and `Field` types.
- **Component-Based Architecture**: Integrates with the `SLM.core` `Component`, `Config`, and `MessageBus` systems.
- **Embedded Documents**: Easily nest documents within one another.
- **Polymorphic Inheritance**: Store different but related document types in the same collection and load them as the correct Python class.
- **Reference Fields**: Create links between documents in different collections.
- **Wrapper-Style Field Access**: Legacy-compatible field manipulation methods
- **Advanced Caching**: Optional field-level caching for performance optimization
- **Factory Methods**: Convenient document creation with new_record()

## Core Components

- `db_component.py`: Contains the unified `MongoODMComponent` that manages connection, registration, and object manager injection.
- `documents.py`: Contains the `Document` and `EmbeddedDocument` base classes.
- `fields.py`: Defines all available field types (`StringField`, `IntField`, `ListField`, etc.).
- `queryset.py`: Implements the query interface (`find`, `find_one`, `save`, `delete`).
- `metaclass.py`: The `ODMBase` metaclass that automatically parses class schemas.

## Quick Start

### 1. Define Your Documents

Create classes that inherit from `Document` or `EmbeddedDocument`.

```python
from SLM.core.mongoODM.documents import Document, EmbeddedDocument
from SLM.core.mongoODM.fields import StringField, IntField, EmbeddedDocumentField

class Address(EmbeddedDocument):
    street = StringField(required=True)
    city = StringField(required=True)

class User(Document):
    __collection__ = "users"  # Required: specifies the MongoDB collection
    name = StringField(required=True)
    email = StringField(unique=True)
    address = EmbeddedDocumentField(Address)
```

### 2. Set Up the Component

In your application's entry point, configure and start the unified `MongoODMComponent`.

```python
from SLM.core.config import Config
from SLM.core.message_bus import MessageBus
from SLM.core.mongoODM.db_component import MongoODMComponent

# a. Configuration
config = Config()
config.set_value("mongodb", {
    "host": "localhost",
    "port": 27017,
    "db_name": "my_app_db"
})

# b. Message Bus (optional)
message_bus = MessageBus()

# c. Unified ODM Component
odm_component = MongoODMComponent(config=config, message_bus=message_bus)

# d. Start the component
odm_component.start() # This connects, registers documents, and creates indexes.
```

### 3. Use the ODM

Once the components are started, you can use the `objects` manager on your `Document` classes to interact with the database.

```python
# Create
user_addr = Address(street="123 Python Lane", city="Codeville")
new_user = User(name="Jane Doe", email="jane.doe@example.com", address=user_addr)
new_user.save()

# Read
found_user = User.objects.find_one({"email": "jane.doe@example.com"})
print(f"Found: {found_user.name}, City: {found_user.address.city}")

# Update
found_user.name = "Jane Smith"
found_user.save()

# Delete
found_user.delete()
```

### 4. Inheritance

To use inheritance, create a base abstract document and subclasses. The ODM will automatically store a `_cls` field in the database to ensure the correct class is loaded.

```python
class Media(Document):
    __collection__ = "media"
    __abstract__ = True
    title = StringField()

class Movie(Media):
    director = StringField()

class Book(Media):
    author = StringField()

# Saving a Movie adds '_cls': 'Movie' to the document
movie = Movie(title="Inception", director="Christopher Nolan").save()

# This query will return a Movie instance
media_item = Media.objects.find_one({"title": "Inception"})
assert isinstance(media_item, Movie)
```

### 5. One-to-Many Relationships

Use a `ReferenceField` on the "many" side and a `ReverseReferenceField` on the "one" side to create a one-to-many link. The `ReverseReferenceField` is a read-only field that provides a query to fetch the related documents.

```python
class Author(Document):
    __collection__ = "authors"
    name = StringField(required=True)
    # This field provides a query for all books by this author
    books = ReverseReferenceField('Book', 'author')

class Book(Media):
    # This field stores a reference to the author
    author = ReferenceField(Author, required=True)
    pages = IntField()

# --- Usage ---
author = Author(name="George Orwell").save()
book1 = Book(title="1984", author=author).save()
book2 = Book(title="Animal Farm", author=author).save()

# The .books attribute is a QuerySet
orwell_books = list(author.books)
assert len(orwell_books) == 2
```

### 6. Indexing

You can define simple indexes directly on your fields for automatic creation when the application starts. Use `index=True` for a standard index and `unique=True` for a unique index.

```python
class User(Document):
    __collection__ = "users"
    # Creates a unique index on the 'email' field
    email = StringField(required=True, unique=True)
    # Creates a standard index on the 'name' field
    name = StringField(required=True, index=True)
```

For more complex scenarios, see the **Advanced Indexing** section below.

### 7. Generic Relationships (Many-to-One)

For situations where a field needs to reference a document in *any* collection, use the `GenericReferenceField`. This is useful for features like reviews, comments, or tags that can apply to different types of documents.

```python
class Book(Document):
    __collection__ = "books"
    title = StringField()

class Movie(Document):
    __collection__ = "movies"
    title = StringField()

class Review(Document):
    __collection__ = "reviews"
    content = StringField()
    # This field can reference a Book, a Movie, or any other Document
    item = GenericReferenceField()

# --- Usage ---
book = Book(title="Dune").save()
movie = Movie(title="Blade Runner").save()

review1 = Review(content="A sci-fi masterpiece.", item=book).save()
review2 = Review(content="Visually stunning.", item=movie).save()

# The reference is resolved automatically
assert isinstance(review1.item, Book)
assert review2.item.title == "Blade Runner"
```

### 8. Advanced Indexing

For compound indexes, text indexes, or other special index types, you can define an `indexes` list within a `Meta` inner class on your document.

The `indexes` list can contain:
- A list of tuples for a standard compound index: `[('field1', 1), ('field2', -1)]`
- A dictionary for more complex indexes, where you can specify a `name` and other options.

```python
from pymongo import TEXT

class Product(Document):
    __collection__ = "products"
    name = StringField()
    category = StringField()
    description = StringField()

    class Meta:
        indexes = [
            # A compound index on name (ascending) and category (descending)
            [('name', 1), ('category', -1)],
            
            # A text index on the description field
            {'fields': [('description', TEXT)], 'name': 'description_text_index'}
        ]
```

The `MongoODMComponent` will automatically detect these definitions and create the corresponding indexes in MongoDB on startup.

## Advanced Features

### 9. Automatic Background Persistence

The mongoODM now supports automatic persistence where field changes are immediately saved to the database without requiring explicit `save()` calls.

```python
class User(Document):
    __collection__ = "users"
    name = StringField()
    email = StringField()

# Create with automatic save
user = User.new_record(name="Alice", email="alice@example.com")
# Document is automatically saved to database

# Changes auto-persist
user.name = "Alice Smith"  # Automatically saved to database
user.email = "alice.smith@example.com"  # Automatically saved

# No manual save() call required!
```

**How it works:**
- The `__setattr__` method is overridden to detect field changes
- Changes are immediately written to MongoDB
- Only the changed fields are updated (efficient partial updates)
- Maintains full backward compatibility with manual save() calls

### 10. Factory Methods and Convenience Creation

Use `new_record()` class method for convenient document creation with automatic persistence:

```python
# Traditional way (still works)
user = User(name="Bob", email="bob@example.com")
user.save()

# New convenient way with auto-save
user = User.new_record(name="Bob", email="bob@example.com")
# Already saved to database!

# Bulk creation
users = [
    User.new_record(name="User1", email="user1@example.com"),
    User.new_record(name="User2", email="user2@example.com"),
    User.new_record(name="User3", email="user3@example.com")
]
# All automatically saved
```

### 11. Wrapper-Style Field Access

For compatibility with legacy systems, the ODM provides wrapper-style field manipulation methods:

```python
class Product(Document):
    __collection__ = "products"
    name = StringField()
    tags = ListField(StringField(), default=[])
    metadata = DictField(default={})

product = Product.new_record(name="Laptop")

# Wrapper-style field access
product.set_field_val("name", "Gaming Laptop")  # Auto-saves
current_name = product.get_field_val("name", "Default Name")

# List manipulation with auto-save
product.list_append("tags", "electronics")  # Auto-saves
product.list_append("tags", "gaming", no_dupes=True)  # Prevents duplicates
product.list_extend("tags", ["portable", "high-performance"])
product.list_remove("tags", "electronics")

# Get list values
tags = product.list_get("tags")  # Returns current list
tag_count = product.list_count("tags")  # Returns count

# Dictionary manipulation
product.dict_set("metadata", "weight", "2.5kg")  # Auto-saves
weight = product.dict_get("metadata", "weight", "Unknown")
product.dict_remove("metadata", "weight")
```

**Available Wrapper Methods:**
- `get_field_val(field_name, default=None)` - Get field value with default
- `set_field_val(field_name, value)` - Set field value with auto-save
- `list_append(field_name, value, no_dupes=False)` - Append to list field
- `list_extend(field_name, values)` - Extend list field
- `list_remove(field_name, value)` - Remove from list field
- `list_get(field_name, default=None)` - Get list field value
- `list_count(field_name)` - Count items in list field
- `dict_set(field_name, key, value)` - Set dictionary key
- `dict_get(field_name, key, default=None)` - Get dictionary value
- `dict_remove(field_name, key)` - Remove dictionary key

### 12. Advanced Caching System

The ODM includes optional field-level caching for performance optimization:

```python
class ExpensiveDocument(Document):
    __collection__ = "expensive_docs"
    name = StringField()
    computed_data = StringField()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable caching for expensive computations
        self._enable_field_caching = True
        self._field_cache = {}

# Usage with caching
doc = ExpensiveDocument.new_record(name="Test")

# First access - computed and cached
value = doc.get_field_val("computed_data")  # Hits database
# Second access - served from cache
value = doc.get_field_val("computed_data")  # From cache

# Cache management
doc._clear_field_cache()  # Clear all cached values
doc._clear_field_cache("computed_data")  # Clear specific field
```

**Caching Features:**
- Optional per-document caching
- Automatic cache invalidation on field changes
- Memory-efficient with manual cache management
- Configurable per field or entire document

### 13. Enhanced Query Interface

Extended query capabilities with additional convenience methods:

```python
class Article(Document):
    __collection__ = "articles"
    title = StringField()
    status = StringField()
    created_at = DateTimeField(default=datetime.utcnow)

# Standard querying (existing)
articles = Article.find({"status": "published"})
article = Article.find_one({"title": "MongoDB Guide"})

# Count without loading documents
count = Article.count({"status": "published"})

# Existence checking
exists = Article.exists({"title": "MongoDB Guide"})

# Get or create pattern
article = Article.get_or_create(
    query={"title": "New Article"},
    defaults={"status": "draft"}
)

# Bulk operations
Article.update_many(
    filter={"status": "draft"},
    update={"$set": {"status": "review"}}
)

Article.delete_many({"status": "archived"})
```

### 14. Migration and Compatibility

The enhanced ODM maintains **100% backward compatibility** while adding new features:

```python
# Old style (still works)
user = User(name="Bob")
user.email = "bob@example.com" 
user.save()  # Manual save required

# New style (automatic persistence)
user = User.new_record(name="Bob", email="bob@example.com")
user.description = "Software Engineer"  # Auto-saves

# Mixed usage
user = User(name="Carol")  # Traditional creation
user.email = "carol@example.com"  # Auto-saves if persistence enabled
user.save()  # Manual save still works
```

**Migration Benefits:**
- **No Breaking Changes** - Existing code continues to work
- **Gradual Adoption** - Can enable automatic persistence per document class
- **Performance Optimization** - Automatic persistence is opt-in
- **Enhanced Debugging** - Better error messages and logging

### 15. Performance and Best Practices

#### Automatic Persistence Performance
```python
# Good: Use wrapper methods for complex operations
user.list_extend("tags", ["python", "mongodb", "odm"])  # Single update

# Avoid: Many individual field changes
user.tag1 = "python"    # Individual update
user.tag2 = "mongodb"   # Individual update  
user.tag3 = "odm"       # Individual update
```

#### Caching Best Practices
```python
# Enable caching for read-heavy documents
class ReadHeavyDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_field_caching = True

# Disable caching for write-heavy documents  
class WriteHeavyDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_field_caching = False
```

#### Indexing for Performance
```python
class OptimizedDocument(Document):
    __collection__ = "optimized"
    
    # Frequently queried fields should be indexed
    email = StringField(unique=True)      # Automatic unique index
    status = StringField(index=True)      # Automatic standard index
    created_at = DateTimeField(index=True)
    
    class Meta:
        indexes = [
            # Compound index for complex queries
            [("status", 1), ("created_at", -1)],
            # Text index for search
            {"fields": [("title", "text"), ("description", "text")]}
        ]
```

## Error Handling and Debugging

The enhanced ODM provides better error handling and debugging capabilities:

```python
from SLM.core.mongoODM.exceptions import ODMException, ValidationError

try:
    user = User.new_record(email="invalid-email")  # Validation error
except ValidationError as e:
    print(f"Validation failed: {e}")

try:
    user = User.new_record(email="existing@example.com")  # Duplicate email
except ODMException as e:
    print(f"ODM error: {e}")

# Enable debug logging for automatic persistence
import logging
logging.getLogger("mongoODM.persistence").setLevel(logging.DEBUG)
```

See `run_examples.py` for a complete, executable demonstration of all features including the new automatic persistence and wrapper-style field access.
