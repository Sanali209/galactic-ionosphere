Below is a **comprehensive** documentation for the **MongoDB Textual Query Parser** code. This parser allows you to write readable, **SQL-like** (or intuitive) queries such as:

```text
role IN ["admin", "superuser"] AND (age >= 18 OR age TYPE int)
```

and automatically convert them into valid MongoDB **filter dictionaries** (no `ParseResults` artifacts) suitable for use with **PyMongo**.  

---

# MongoDB Textual Query Parser Documentation

## Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Example Queries](#example-queries)  
4. [Installation & Dependencies](#installation--dependencies)  
5. [Usage](#usage)  
6. [Detailed Explanation](#detailed-explanation)  
   - [Grammar](#grammar)  
   - [Normalization Step](#normalization-step)  
7. [API Reference](#api-reference)  
   - [`parse_mongo_query(query_str)`](#parse_mongo_query)  
   - [`parse_mongo_query_with_normalization(query_str)`](#parse_mongo_query_with_normalization)  
8. [Extending the Parser](#extending-the-parser)  
9. [License](#license)  

---

## Introduction

When working with MongoDB in Python, queries are typically built with dictionaries, e.g. `{'age': {'$gt': 30}}`. For **complex** logical expressions with `$and`, `$or`, `$not`, and specialized operators (`$in`, `$exists`, `$regex`, etc.), **managing the raw dictionaries can become cumbersome**.

This **textual query parser** provides a **pyparsing**-based DSL (Domain-Specific Language) so you can write more readable query strings—resembling a simplified SQL or natural language—and then convert them to **MongoDB filter** dictionaries automatically.

---

## Features

1. **Logical Operators**:  
   - **AND** → `$and`  
   - **OR** → `$or`  
   - **NOT** → `$not`  

2. **Comparison Operators**:  
   - `==`  → `$eq`  
   - `!=`  → `$ne`  
   - `>`   → `$gt`  
   - `>=`  → `$gte`  
   - `<`   → `$lt`  
   - `<=`  → `$lte`  

3. **Set/Array Operators**:  
   - `IN` / `NOT IN` → `$in` / `$nin`  
   - `ALL` → `$all`  
   - `SIZE` → `$size`  

4. **Other MongoDB Operators**:  
   - `EXISTS` → `$exists`  
   - `REGEX`  → `$regex`  
   - `TYPE`   → `$type`  
   - `MOD`    → `$mod`  

5. **Arrays**: queries like `tags IN ["python", "mongodb"]`  

6. **Booleans**: parse `true` / `false` as Python `True` / `False`.  

7. **Parentheses**: `( ... )` sub-expressions for complex nested logic.  

8. **Normalization Step**: ensures no leftover pyparsing `ParseResults`, so final data is **pure Python dicts, lists, strings, ints, floats, booleans**.

---

## Example Queries

- **Simple**:
  ```
  age >= 18
  ```
- **Using IN**:
  ```
  role IN ["admin", "superuser"]
  ```
- **Combining AND, OR**:
  ```
  role IN ["admin", "superuser"] AND (age >= 18 OR age TYPE int)
  ```
- **Using NOT**:
  ```
  NOT city EXISTS AND tags SIZE 3
  ```
- **Regex**:
  ```
  title REGEX "^MongoDB"
  ```

Each of these is parsed into a **MongoDB filter** dictionary, for instance:

```python
"NOT city EXISTS AND tags SIZE 3"

# => {
#   "$and": [
#       {"city": {"$not": {"$exists": True}}},
#       {"tags": {"$size": 3}}
#   ]
# }
```

---

## Installation & Dependencies

1. **Python 3.6+** recommended.  
2. Install **pyparsing** (and possibly `pyparsing_common`):
   ```bash
   pip install pyparsing
   ```

3. Place the parser code (shown below) in a file, for example, `mongoQueryTextParser.py`.

---

## Usage

1. **Import** the desired function:
   ```python
   from mongoQueryTextParser import parse_mongo_query_with_normalization
   ```
2. **Parse** a textual query:
   ```python
   query_str = 'role IN ["admin", "superuser"] AND (age >= 18 OR age TYPE int)'
   mongo_filter = parse_mongo_query_with_normalization(query_str)
   print(mongo_filter)
   # {
   #   "$and": [
   #     {"role": {"$in": ["admin", "superuser"]}},
   #     {
   #       "$or": [
   #         {"age": {"$gte": 18}},
   #         {"age": {"$type": "int"}}
   #       ]
   #     }
   #   ]
   # }
   ```
3. **Use** the resulting dictionary in PyMongo:
   ```python
   results = my_collection.find(mongo_filter)
   for doc in results:
       print(doc)
   ```

---

## Detailed Explanation

### Grammar

- **Field Name**: letters, digits, underscores, dots.
- **Operators**:  
  - **Logical**: `AND`, `OR`, `NOT`  
  - **Comparison**: `==`, `!=`, `>`, `>=`, `<`, `<=`  
  - **Array**: `IN`, `NOT IN`, `ALL`, `SIZE`  
  - **Other**: `REGEX`, `TYPE`, `EXISTS`, `MOD`  
- **Values**:  
  - Single/double-quoted strings,  
  - Integers/floats,  
  - Booleans (`true` / `false`),  
  - Arrays like `[val1, val2, ...]`.

We define a **single condition** as a **field + operator + optional value** (e.g. `age >= 18`). We then combine conditions using **pyparsing**’s `infixNotation` to implement `( ... )`, `NOT`, `AND`, `OR` in an expression tree.

### Normalization Step

Pyparsing sometimes returns nested `ParseResults`. We apply a final `_normalize` function to recursively convert `ParseResults` into plain Python objects. This ensures the final filter is strictly a Python `dict`, `list`, `str`, `int`, `float`, or `bool`.

---

## API Reference

### `parse_mongo_query(query_str)`

> **Definition**  
> ```python
> def parse_mongo_query(query_str: str) -> dict:
>     ...
> ```
> **Description**  
> Parses the query string using the textual grammar and returns a MongoDB filter dictionary.  
> **Note**: This version by default is configured to eliminate many `ParseResults`, but if any remain, see `parse_mongo_query_with_normalization`.

#### Example
```python
from mongoQueryTextParser import parse_mongo_query

query_str = 'category NOT IN ["food","drink"]'
mongo_filter = parse_mongo_query(query_str)
print(mongo_filter)
# => {"category": {"$nin": ["food","drink"]}}
```
---

### `parse_mongo_query_with_normalization(query_str)`

> **Definition**  
> ```python
> def parse_mongo_query_with_normalization(query_str: str) -> dict:
>     ...
> ```
> **Description**  
> Same as `parse_mongo_query`, but applies a final `_normalize(...)` pass, guaranteeing **no** leftover `ParseResults`. If you notice any parse artifacts, use this function.

#### Example
```python
from mongoQueryTextParser import parse_mongo_query_with_normalization

query_str = 'NOT city EXISTS AND tags SIZE 3'
mongo_filter = parse_mongo_query_with_normalization(query_str)
print(mongo_filter)
# => {
#   '$and': [
#       {'city': {'$not': {'$exists': True}}},
#       {'tags': {'$size': 3}}
#   ]
# }
```

---

## Extending the Parser

You can **add** more MongoDB operators (like `$elemMatch`, `$where`, `$geoWithin`) by extending:

1. **Grammar**: include keywords like `ELEMMATCH`, `WHERE`, etc.
2. **`_make_single_condition(...)`**: map the new textual operator to a MongoDB operator, e.g. `ELEMMATCH -> "$elemMatch"`.

For example:
```python
# Add a new line in comparison_ops:
pp.Keyword("ELEMMATCH", caseless=True),

# In _make_single_condition:
if op_upper == "ELEMMATCH":
    return {field: {"$elemMatch": value}}
```

---

## License

This code is provided under an open-source license (e.g. **MIT**). Feel free to copy, modify, or integrate it into your own projects. If you publish derivative works, consider preserving this documentation and giving credit to contributors.


**That’s it!** You can now use this parser to handle textual queries in a more **readable** manner and pass them to your MongoDB driver (PyMongo) seamlessly.