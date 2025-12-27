## MongoDBQueryParser Documentation

### Overview

The `MongoDBQueryParser` class is designed to parse human-readable query strings into MongoDB-compatible query dictionaries. This allows for the creation of complex MongoDB queries using a simplified syntax.

### Use Cases

- **Text Search**: Convert simple search phrases into MongoDB text search queries.
- **Comparison Queries**: Parse queries with equality, inequality, and range conditions.
- **Logical Queries**: Handle logical conditions like AND, OR, and nested parentheses.
- **Array and Regex Queries**: Support for array membership and regular expression matching.

### Features

1. **Equality** (`=`): Matches documents where the field equals the specified value.
2. **Inequality** (`!=`): Matches documents where the field does not equal the specified value.
3. **Less Than** (`<`): Matches documents where the field is less than the specified value.
4. **Greater Than** (`>`): Matches documents where the field is greater than the specified value.
5. **Less Than or Equal To** (`<=`): Matches documents where the field is less than or equal to the specified value.
6. **Greater Than or Equal To** (`>=`): Matches documents where the field is greater than or equal to the specified value.
7. **In** (`in`): Matches documents where the field's value is in the specified list.
8. **Not In** (`nin`): Matches documents where the field's value is not in the specified list.
9. **Regex** (`$regex`): Matches documents where the field's value matches the specified regular expression.
10. **Text Search** (`$text`): Matches documents using MongoDB's text search feature.
11. **Exists** (`$exist`): Matches documents where the field exists or does not exist.



### Examples of Use Cases

1. **Text Search**: Converts a simple search phrase into a MongoDB text search query.
    ```python
    query_string = "MongoDB tutorial"
    query = parser.parse_query(query_string)
    # Result: {"$text": {"$search": "MongoDB tutorial"}}
    ```

2. **Equality**: Parses a query for equality conditions.
    ```python
    query_string = "name = 'Alice'"
    query = parser.parse_query(query_string)
    # Result: {"name": "Alice"}
    ```

3. **Inequality**: Parses a query for inequality conditions.
    ```python
    query_string = "age != 25"
    query = parser.parse_query(query_string)
    # Result: {"age": {"$ne": 25}}
    ```

4. **Range Conditions**: Parses queries for range conditions like less than and greater than.
    ```python
    query_string = "age > 20 and age < 30"
    query = parser.parse_query(query_string)
    # Result: {"$and": [{"age": {"$gt": 20}}, {"age": {"$lt": 30}}]}
    ```

5. **Array Membership**: Parses queries for checking array membership.
    ```python
    query_string = "tags in ['mongodb', 'database']"
    query = parser.parse_query(query_string)
    # Result: {"tags": {"$in": ["mongodb", "database"]}}
    ```

6. **Regular Expressions**: Parses queries for regular expression matching.
    ```python
    query_string = "name $regex '^A.*'"
    query = parser.parse_query(query_string)
    # Result: {"name": {"$regex": "^A.*"}}
    ```

7. **Existence Check**: Parses queries to check if a field exists or not.
    ```python
    query_string = "email $exist true"
    query = parser.parse_query(query_string)
    # Result: {"email":