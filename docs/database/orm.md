# Mongo ORM (`src.core.database`)

An Async, Reactive Object-Relational Mapper built on top of `motor`.

## Features
-   **AsyncIO**: Fully non-blocking operations.
-   **Polymorphism**: Automatic Single Table Inheritance.
-   **Reactivity**: Events emitted on field modifications.
-   **Validation & Conversion**: Typed fields with custom logic.
-   **Complex Types**: Helpers for Lists and Dictionaries.

## Defining Models

```python
from src.core.database.orm import CollectionRecord, FieldPropInfo

def validate_age(val):
    return 0 <= val <= 120

class User(CollectionRecord, table="users",indexes=[
    "name",                   # Simple index
    [("age", 1), ("name", -1)] # Compound index (multikey/sort)
]):
    # Simple Field
    name = FieldPropInfo("name", default="Guest", field_type=str)
    
    # Validated Field
    age = FieldPropInfo("age", default=0, field_type=int, validator=validate_age)
    
    # Converted Field (e.g. storage as string, usage as list)
    tags = FieldPropInfo("tags", default=[], field_type=list)

# Polymorphism: Admin is stored in 'users' with _cls='Admin'
class Admin(User):
    permissions = FieldPropInfo("perms", default=[], field_type=list)
```

## Advanced Usage

### 1. CRUD & Async

```python
# Create
user = User(name="Alice", age=30)
await user.save() # Insert

# Read
user = await User.get("507f1f77bcf86cd799439011")

# Update
user.age = 31 # Triggers on_change
await user.save() # Update

# Delete
await user.delete()
```

### 2. Reactivity

Listen to changes on specific instances.

```python
def on_user_change(obj, field, value):
    print(f"User {obj.id} changed '{field}' to '{value}'")

user = User()
user.on_change.connect(on_user_change)

user.name = "Bob" # Output: User ... changed 'name' to 'Bob'
```

### 3. Polymorphism

You can mix different subclasses in the same collection. The ORM automatically handles the `_cls` field.

```python
# Save an Admin
admin = Admin(name="Root", permissions=["all"])
await admin.save()

# Querying the Base Class returns the correct Subclass
user_id = admin.id
loaded_obj = await User.get(user_id)

print(type(loaded_obj)) # <class 'Admin'>
print(loaded_obj.permissions) # ['all']
```

### 4. Working with Lists and Dicts

Since `FieldPropInfo` only detects assignment (`self.x = y`), modifying mutable objects in-place (like `list.append`) won't trigger events or update the internal cache correctly unless handled carefully.

**Use the helper methods to ensure reactivity:**

```python
# List Append
user.list_append("tags", "new_tag") # Triggers on_change("tags", [...])

# Dict Update
user.dict_update("metadata", "last_login", "2024-01-01") 
```

### 5. Validation and Converters

Field definitions can include logic that runs on assignment.

```python
def to_upper(val):
    return val.upper() if isinstance(val, str) else val

class Item(CollectionRecord, table="items"):
    # Auto-converts to Uppercase
    code = FieldPropInfo("code", converter=to_upper)
    
item = Item()
item.code = "abc"
print(item.code) # "ABC"
```
