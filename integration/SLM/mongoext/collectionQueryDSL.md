# MongoDB DSL Documentation

This document serves as a complete reference for a Python-based **Domain-Specific Language (DSL)** that simplifies the creation and execution of MongoDB queries. By leveraging Python's operator overloading and method chaining, the DSL allows for more readable and maintainable code when constructing complex MongoDB queries.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Key Concepts & Overview](#key-concepts--overview)  
3. [Installation & Setup](#installation--setup)  
4. [Usage Examples](#usage-examples)  
5. [API Reference](#api-reference)  
   - [Condition Class](#condition-class)  
   - [Field (Q) Class](#field-q-class)  
   - [QueryBuilder Class](#querybuilder-class)  
   - [DSL Class](#dsl-class)  
6. [Common MongoDB Operators](#common-mongodb-operators)  
7. [Extending the DSL](#extending-the-dsl)  
8. [License](#license)

---

## Introduction

When working with MongoDB in Python, constructing queries using raw dictionaries can quickly become unwieldy, especially for complex queries. This DSL provides a more Pythonic, fluent interface to build MongoDB queries, featuring:

- **Logical operators** (`&`, `|`, `~` for AND, OR, NOT).
- **Comparison operators** (`eq`, `ne`, `lt`, `lte`, `gt`, `gte`).
- **Array operators** (`all_`, `size`, `elemMatch`).
- **Misc operators** (`exists`, `in_`, `nin`, `regex`, `type`, `mod`, `whereJs`).
- **Query builder** for combining filter criteria, projection, sorting, skipping, limiting, and direct execution with a PyMongo collection.
- Easy integration with PyMongo methods like `find`, `count_documents`, `delete_many`, and `update_many`.

---

## Key Concepts & Overview

### Chaining & Fluent Interface

Instead of writing verbose dictionary-based queries, the DSL allows constructing queries like:

```python
Q("age").gte(21) & Q("city").eq("London")
```

and combining conditions:

```python
(Q("age").gte(21) & Q("city").eq("London")) | Q("role").eq("admin")
```

### Logical Operators

- **`&`** : Logical AND  
- **`|`** : Logical OR  
- **`~`** : Logical NOT  

```python
Q("city").eq("London") & Q("status").exists(True)
```

yields a condition similar to:

```python
{"$and": [{"city": "London"}, {"status": {"$exists": True}}]}
```

### Usage Flow

1. Create conditions using `Q("field")` (short for `Field("field")`).
2. Combine conditions with logical operators.
3. Pass final condition to `DSL.find(...)`.
4. Optionally specify:
   - `.projection(...)`
   - `.sort(...)`
   - `.skip(...)`
   - `.limit(...)`
5. Call `.execute()` to get a PyMongo Cursor or use `.count()`, `.delete_many()`, `.update_many(...)` for other operations.

---

## Installation & Setup

1. **Python 3.6+** is recommended.
2. **PyMongo** is required (`pip install pymongo`).
3. Copy or include the DSL Python file(s) in your project.
4. Import the necessary classes:

```python
from my_dsl_file import DSL, Q
```

5. Instantiate the DSL with your **PyMongo collection** reference:

```python
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["my_database"]
collection = db["my_collection"]

dsl = DSL(collection)
```

---

## Usage Examples

### Basic Find with Projection

```python
# Example: Find documents where age is >= 18
query = Q("age").gte(18)

results = (
    dsl.find(query)
       .projection(["name", "age"])
       .execute()
)

for doc in results:
    print(doc)
```

### Complex Logical Query

```python
# Example: ((age >= 21 AND city="London") OR role="admin") AND tags array has length=3

query = (
    (
        (Q("age").gte(21) & Q("city").eq("London"))
        | Q("role").eq("admin")
    )
    & Q("tags").size(3)
)

results = (
    dsl.find(query)
       .projection(["name", "city", "role", "tags"])
       .sort("age", descending=True)
       .limit(10)
       .execute()
)

for doc in results:
    print(doc)
```

### Counting Documents

```python
count_admins = dsl.find(Q("role").eq("admin")).count()
print("Number of admin users:", count_admins)
```

### Updating Documents

```python
update_result = dsl.find(Q("role").eq("guest")).update_many({"$set": {"role": "user"}})
print(f"Modified count: {update_result.modified_count}")
```

### Deleting Documents

```python
delete_result = dsl.find(Q("obsolete").eq(True)).delete_many()
print(f"Deleted count: {delete_result.deleted_count}")
```

---

## API Reference

Below is an in-depth reference for each class and its methods.

### Condition Class

```python
class Condition:
    def __init__(self, query: Dict[str, Any] = None)
    def __and__(self, other: "Condition") -> "Condition"
    def __or__(self, other: "Condition") -> "Condition"
    def __invert__(self) -> "Condition"
```

- **`Condition`** objects hold partial or complete MongoDB queries as Python dictionaries.
- **`__and__`, `__or__`, `__invert__`** implement logical operators (`&`, `|`, `~`).
- Typically, you do not instantiate `Condition` directly; they are created when you call methods on a `Field` or `Q`.

**Examples**:
```python
cond1 = Q("age").gte(21)  # => Condition({"age": {"$gte": 21}})
cond2 = Q("city").eq("London")
combined = cond1 & cond2   # => Condition({"$and": [..., ...]})
```

---

### Field (Q) Class

```python
def Q(field_name: str) -> "Field"
class Field:
    def eq(self, value: Any) -> Condition
    def ne(self, value: Any) -> Condition
    def lt(self, value: Any) -> Condition
    def lte(self, value: Any) -> Condition
    def gt(self, value: Any) -> Condition
    def gte(self, value: Any) -> Condition
    def in_(self, values: List[Any]) -> Condition
    def nin(self, values: List[Any]) -> Condition
    def exists(self, does_exist: bool = True) -> Condition
    def regex(self, pattern: str, options: str = "") -> Condition
    def type(self, bson_type: Union[int, str]) -> Condition
    def mod(self, divisor: int, remainder: int) -> Condition
    def whereJs(self, js_expr: str) -> Condition
    def all_(self, values: List[Any]) -> Condition
    def size(self, size_val: int) -> Condition
    def elemMatch(self, condition: Condition) -> Condition
```

- **`Q(field_name)`** is a convenience function returning `Field(field_name)`.
- Each method returns a `Condition` that can be combined with others.

**Examples**:

- `Q("age").eq(18)` -> `{"age": 18}`
- `Q("age").gte(21)` -> `{"age": {"$gte": 21}}`
- `Q("tags").all_(["mongodb", "python"])` -> `{"tags": {"$all": ["mongodb", "python"]}}`
- `Q("status").exists(True)` -> `{"status": {"$exists": True}}`

---

### QueryBuilder Class

```python
class QueryBuilder:
    def __init__(
        self,
        base_query: Dict[str, Any] = None,
        projection: Union[Dict[str, int], List[str], None] = None,
        sort: List[Tuple[str, int]] = None,
        skip_count: int = 0,
        limit_count: int = 0
    )

    def find(self, condition: Condition) -> "QueryBuilder"
    def projection(self, fields: Union[List[str], Dict[str, int]]) -> "QueryBuilder"
    def sort(self, field: str, ascending: bool = True, descending: bool = False) -> "QueryBuilder"
    def skip(self, count: int) -> "QueryBuilder"
    def limit(self, count: int) -> "QueryBuilder"
    def build(self) -> Dict[str, Any]
```

- Internally used by the `DSL` class but can be used directly if desired.
- **`find(condition)`** replaces the existing query with `condition.query`.
- **`projection(fields)`** sets fields to include/exclude.
- **`sort(field, ascending=True, descending=False)`** appends a field to the sort list.
  - If `descending=True`, sorts by -1.
  - Otherwise, by default, sorts ascending with +1.
- **`skip(count)`** sets skip value.
- **`limit(count)`** sets limit value.
- **`build()`** returns a dictionary suitable for passing to `pymongo.collection.Collection.find()` with `**kwargs`.

**Example**:
```python
qb = QueryBuilder()
qb = qb.find(Q("age").gte(21))
qb = qb.projection(["name", "age"]).sort("age", descending=True).limit(10)
final_query_spec = qb.build()
# final_query_spec => {
#   'filter': {'age': {'$gte': 21}},
#   'projection': {'name': 1, 'age': 1},
#   'sort': [('age', -1)],
#   'limit': 10
# }
```

---

### DSL Class

```python
class DSL:
    def __init__(self, collection: Collection)
    def find(self, condition: Condition) -> "DSL"
    def projection(self, fields: Union[List[str], Dict[str, int]]) -> "DSL"
    def sort(self, field: str, ascending: bool = False, descending: bool = False) -> "DSL"
    def skip(self, count: int) -> "DSL"
    def limit(self, count: int) -> "DSL"
    def execute(self)
    def count(self) -> int
    def delete_many(self)
    def update_many(self, update_doc: Dict[str, Any])
    def build_query(self) -> Dict[str, Any]
```

- **`DSL`** is the high-level interface for building queries **and** executing them against a PyMongo `collection`.
- Each method returns a new `DSL` instance (immutable style) so you can chain calls.
- **`execute()`** calls `collection.find(**params)` internally and returns a PyMongo Cursor.
- **`count()`** calls `collection.count_documents(self._qb.base_query)`.
- **`delete_many()`** calls `collection.delete_many(self._qb.base_query)`.
- **`update_many(update_doc)`** calls `collection.update_many(self._qb.base_query, update_doc)`.
- **`build_query()`** returns a dictionary for debugging or direct usage in `collection.find(**dict)`.

**Typical usage**:
```python
dsl = DSL(collection)
results = (
    dsl.find(Q("active").eq(True))
       .projection(["name", "email"])
       .sort("name", descending=False)
       .skip(0)
       .limit(50)
       .execute()
)

for doc in results:
    print(doc)
```

---

## Common MongoDB Operators

This DSL covers a broad set of operators:

- **Comparison**: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`
- **Logical**: `$and`, `$or`, `$not`
- **Array**: `$all`, `$size`, `$elemMatch`
- **Existence**: `$exists`
- **Inclusion**: `$in`, `$nin`
- **Pattern**: `$regex`
- **Type**: `$type`
- **Mod**: `$mod`
- **Where (JavaScript)**: `$where`

You can further extend this code to handle advanced operators like `$geoNear`, `$text`, `$search`, etc.

---

## Extending the DSL

This DSL is designed with extensibility in mind. You can add more methods to `Field` or more wrappers in the `DSL` class. For example, to add `$text` search support:

```python
def text_search(self, search_str: str) -> Condition:
    return Condition({"$text": {"$search": search_str}})
```

Then, use it like:

```python
query = Q("any_field").text_search("mongodb DSL")
```

---

## License

This DSL code is provided under an open-source license (you can specify **MIT**, **Apache 2.0**, or any license you prefer). Feel free to adapt or modify it for your project.  

---

### Final Thoughts

This DSL aims to make writing MongoDB queries in Python more natural, expressive, and maintainable. By leveraging Python’s features like operator overloading and chaining, developers can write complex queries more concisely and in a more readable manner.  

For any issues or further improvements, feel free to open pull requests or issues in your project’s repository!