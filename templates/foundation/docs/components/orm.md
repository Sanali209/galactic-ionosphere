# Async MongoDB ORM

The Foundation Template includes a lightweight, asynchronous Object-Document Mapper (ODM) built on top of `motor`.

## Features
- **AsyncIO**: Native async/await support
- **Auto Collection Naming**: Automatically generates collection names from class names
- **Typing**: Pydantic-style field descriptors with full control
- **Relationships**: Declarative 1:1, 1:N, and M:N references
- **Embedding**: Support for nested documents and lists
- **Indexing**: Declarative index generation
- **Reactive**: ObserverEvent integration for field changes

## Defining Models

Models inherit from `CollectionRecord`. Collection names are auto-generated from class names.

```python
from foundation import (
    CollectionRecord, Field,
    ReferenceField, ListField, EmbeddedField, DictField
)

# Auto collection name: "addresses" (if used standalone)
class Address(CollectionRecord):
    street: str = Field(default="")
    city: str = Field(default="")
    zip_code: str = Field(default="")

# Auto collection name: "users"
class User(CollectionRecord):
    # Simple fields
    username: str = Field(default="")
    email: str = Field(default="")
    role: str = Field(default="user")
    
    # Advanced fields with index/unique constraints
    user_id: str = Field(default="", index=True, unique=True)
    
    # Embedded document
    address: Address = EmbeddedField(Address)
    
    # Lists
    tags: list = ListField(Field())

# Override auto-naming if needed
class CustomCollection(CollectionRecord):
    _collection_name = "my_custom_name"  # Override
    name: str = Field(default="")
```

### Auto Collection Naming Rules

| Class Name | Auto Collection Name |
|-----------|---------------------|
| `User` | `users` |
| `ImageRecord` | `image_records` |
| `SearchHistory` | `search_histories` |
| `HTTPResponse` | `http_responses` |

**Pluralization logic:**
- Ends with 'y' → replace with 'ies' (History → histories)
- Ends with 's' → add 'es' (Address → addresses)
- Default → add 's' (User → users)

## Relationships

Use `ReferenceField` to link documents.

```python
class Post(CollectionRecord):
    # Auto collection: "posts"
    title: str = Field(default="")
    content: str = Field(default="")
    author: User = ReferenceField(User)  # Reference to User
    
    # Many-to-many via list of references
    tags: list = ListField(ReferenceField(Tag))
```

## CRUD Operations

### Creating & Saving

```python
# Create
user = User(username="alice", email="alice@example.com")
user.tags = ["admin", "developer"]
await user.save()

# Create with relationship
post = Post(title="Hello World", author=user)
await post.save()
```

### Querying

```python
# Find one
user = await User.find_one({"username": "alice"})

# Find many with sort and limit
recent_posts = await Post.find(
    {"author": user._id},
    sort=[("created_at", -1)],
    limit=10
)

# Get by ID
user = await User.get(some_object_id)
```

### Resolving References

```python
# Load a post
post = await Post.find_one({"title": "Hello World"})

# Resolve reference (lazy loading)
author = await post.author.fetch()
print(author.username)  # "alice"

# Bulk fetch for lists
tags = await post.tags.fetch_all()
```

### Updating

```python
user = await User.find_one({"username": "alice"})
user.role = "admin"
await user.save()  # Upserts based on _id
```

### Deleting

```python
user = await User.find_one({"username": "alice"})
await user.delete()
```

## Indexing

Indexes are defined declaratively on fields or in `_indexes` for compound keys.

```python
class User(CollectionRecord):
    # Single field index
    email: str = Field(default="", index=True, unique=True)
    username: str = Field(default="", index=True)
    
    # Compound index (define in metaclass)
    _indexes = [
        [("email", 1), ("username", 1)]  # Compound index
    ]

# Apply indexes to database
await User.ensure_indexes()
```

## Field Types

Available field types from `foundation`:

```python
from foundation import (
    Field,           # Base field (works for any type)
    StringField,     # String specific (currently alias to Field)
    IntField,        # Integer specific (currently alias to Field)
    BoolField,       # Boolean specific (currently alias to Field)
    DictField,       # Dictionary with default factory
    ReferenceField,  # Reference to another CollectionRecord
    EmbeddedField,   # Embedded document
    ListField,       # List of fields
)
```

### Field Options

```python
Field(
    default=None,           # Default value or callable factory
    index=False,            # Create database index
    unique=False,           # Unique constraint
    # Future: required, validator, etc.
)
```

## Reactive Changes

Models emit change events via `ObserverEvent`:

```python
user = User(username="alice")

def on_change(record, field_name, new_value):
    print(f"{field_name} changed to {new_value}")

user.on_change.subscribe(on_change)
user.username = "alice_updated"  # Triggers event
```

## Best Practices

1. **Use auto-collection naming** unless you have legacy data
2. **Always use Field()** for control over index/unique/validation
3. **Call `ensure_indexes()`** during app startup
4. **Use references** for relationships, not embedded IDs
5. **Leverage `fetch_all()`** for efficient bulk loading

## Example: Complete Model

```python
from foundation import CollectionRecord, Field, ReferenceField, ListField
from datetime import datetime

class BlogPost(CollectionRecord):
    # Auto collection: "blog_posts"
    
    title: str = Field(default="", index=True)
    slug: str = Field(default="", index=True, unique=True)
    content: str = Field(default="")
    published: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Relationships
    author: User = ReferenceField(User)
    tags: list = ListField(ReferenceField(Tag))
    
    def __str__(self):
        return f"Post: {self.title}"

# Usage
post = BlogPost(
    title="Getting Started",
    slug="getting-started",
    author=some_user
)
await post.save()
await BlogPost.ensure_indexes()
```

