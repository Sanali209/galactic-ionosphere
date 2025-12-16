# Async MongoDB ORM

The Foundation Template includes a lightweight, asynchronous Object-Document Mapper (ODM) built on top of `motor`.

## Features
- **AsyncIO**: Native async/await support.
- **Typing**: Pydantic-style field descriptors.
- **Relationships**: Declarative 1:1, 1:N, and M:N references.
- **Embedding**: Support for nested documents and lists.
- **Indexing**: Declarative index generation.

## Defining Models

Models must inherit from `CollectionRecord` and define `_collection_name`.

```python
from src.core.database.orm import (
    CollectionRecord, StringField, IntField, 
    ReferenceField, ListField, EmbeddedField, DictField
)

class Address(CollectionRecord):
    # Embedded models don't need _collection_name if only used as fields
    street = StringField()
    city = StringField()

class User(CollectionRecord):
    _collection_name = "users"
    
    # Fields
    username = StringField(index=True, unique=True)
    email = StringField()
    role = StringField(default="user")
    
    # Embedding
    address = EmbeddedField(Address)
    
    # Lists
    tags = ListField(StringField())
```

## Relationships

Use `ReferenceField` to link documents.

```python
class Post(CollectionRecord):
    _collection_name = "posts"
    title = StringField()
    author = ReferenceField(User)
```

### Saving & Loading

```python
# Create
user = User(username="alice", tags=["admin"])
await user.save()

# Create Related
post = Post(title="Hello", author=user)
await post.save()

# Query
found_post = await Post.find_one({"title": "Hello"})

# Resolve Reference
# Accessing .author returns a Reference wrapper or the ID
# Call fetch() to get the actual object
author_obj = await found_post.author.fetch()
print(author_obj.username)
```

## Indexing

Indexes are defined declaratively on fields (`index=True`) or in the `_indexes` list for compound keys.

```python
# To apply indexes to the DB:
await User.ensure_indexes()
await Post.ensure_indexes()
```
