from typing import Any, Dict, List, Tuple, Union
from pymongo.collection import Collection


class Condition:
    """
    Represents a MongoDB condition (query fragment).
    Allows chaining of operators and combining with & and |.
    """

    def __init__(self, query: Dict[str, Any] = None):
        self.query = query if query is not None else {}

    def __and__(self, other: "Condition") -> "Condition":
        """
        Logical AND
        """
        return Condition({
            "$and": [
                self.query,
                other.query
            ]
        })

    def __or__(self, other: "Condition") -> "Condition":
        """
        Logical OR
        """
        return Condition({
            "$or": [
                self.query,
                other.query
            ]
        })

    def __invert__(self) -> "Condition":
        """
        Logical NOT
        """
        return Condition({
            "$not": self.query
        })

    def __repr__(self) -> str:
        return f"Condition({self.query})"


def Q(field_name: str) -> "Field":
    """
    Shortcut to create a new Field object.
    """
    return Field(field_name)


class Field:
    """
    Represents a field in a MongoDB document. Allows building conditions
    like: Q("age").gte(18), Q("tags").in_(["mongo", "python"])
    """

    def __init__(self, field_name: str):
        self.field_name = field_name

    def eq(self, value: Any) -> Condition:
        return Condition({self.field_name: value})

    def ne(self, value: Any) -> Condition:
        return Condition({self.field_name: {"$ne": value}})

    def lt(self, value: Any) -> Condition:
        return Condition({self.field_name: {"$lt": value}})

    def lte(self, value: Any) -> Condition:
        return Condition({self.field_name: {"$lte": value}})

    def gt(self, value: Any) -> Condition:
        return Condition({self.field_name: {"$gt": value}})

    def gte(self, value: Any) -> Condition:
        return Condition({self.field_name: {"$gte": value}})

    def in_(self, values: List[Any]) -> Condition:
        return Condition({self.field_name: {"$in": values}})

    def nin(self, values: List[Any]) -> Condition:
        return Condition({self.field_name: {"$nin": values}})

    def exists(self, does_exist: bool = True) -> Condition:
        return Condition({self.field_name: {"$exists": does_exist}})

    def regex(self, pattern: str, options: str = "") -> Condition:
        """
        pattern: Regex pattern
        options: e.g. "i" for case-insensitive
        """
        if options:
            return Condition({
                self.field_name: {
                    "$regex": pattern,
                    "$options": options
                }
            })
        else:
            return Condition({
                self.field_name: {"$regex": pattern}
            })

    def type(self, bson_type: Union[int, str]) -> Condition:
        """
        Matches documents where the field is of the specified BSON type.
        Example: Q("field").type("string") or Q("field").type(2)
        """
        return Condition({self.field_name: {"$type": bson_type}})

    def mod(self, divisor: int, remainder: int) -> Condition:
        """
        field % divisor == remainder
        """
        return Condition({self.field_name: {"$mod": [divisor, remainder]}})

    def whereJs(self, js_expr: str) -> Condition:
        """
        Execute a JavaScript expression on the server side.
        Example usage:
          Q("this").whereJs("this.fieldName == 'someValue'")
        """
        return Condition({"$where": js_expr})

    # -------- Array operators -----------
    def all_(self, values: List[Any]) -> Condition:
        """
        Matches arrays that contain all elements specified.
        """
        return Condition({self.field_name: {"$all": values}})

    def size(self, size_val: int) -> Condition:
        """
        Matches arrays with the specified number of elements.
        """
        return Condition({self.field_name: {"$size": size_val}})

    def elemMatch(self, condition: Condition) -> Condition:
        """
        Matches documents that contain an array field with at least one
        element matching the specified condition.
        """
        return Condition({self.field_name: {"$elemMatch": condition.query}})


class QueryBuilder:
    """
    Manages how to build and store query, projection, sort, limit, skip, etc.
    """

    def __init__(self,
                 base_query: Dict[str, Any] = None,
                 projection: Union[Dict[str, int], List[str], None] = None,
                 sort: List[Tuple[str, int]] = None,
                 skip_count: int = 0,
                 limit_count: int = 0):
        self.base_query = base_query if base_query is not None else {}
        self.projection_fields = projection
        self.sort_fields = sort if sort else []
        self.skip_count = skip_count
        self.limit_count = limit_count

    def find(self, condition: Condition) -> "QueryBuilder":
        """
        Replace the base query with the condition's query.
        """
        qb = self._clone()
        qb.base_query = condition.query
        return qb

    def projection(self, fields: Union[List[str], Dict[str, int]]) -> "QueryBuilder":
        """
        Specify which fields to include/exclude.
        If passing a list of strings, it includes those fields.
        If passing a dict, it uses the dict as-is.
        """
        qb = self._clone()
        if isinstance(fields, list):
            qb.projection_fields = {field: 1 for field in fields}
        else:
            qb.projection_fields = fields
        return qb

    def sort(self, field: str, ascending: bool = True, descending: bool = False) -> "QueryBuilder":
        """
        Add a sort field. ascending=True or descending=True
        """
        qb = self._clone()
        direction = -1 if descending else 1
        qb.sort_fields.append((field, direction))
        return qb

    def skip(self, count: int) -> "QueryBuilder":
        qb = self._clone()
        qb.skip_count = count
        return qb

    def limit(self, count: int) -> "QueryBuilder":
        qb = self._clone()
        qb.limit_count = count
        return qb

    def build(self) -> Dict[str, Any]:
        """
        Build a dictionary of the full query configuration:
         {
           "filter": {...},
           "projection": {...} or None,
           "sort": [...],
           "skip": ...,
           "limit": ...
         }
        This structure can be directly passed to PyMongo methods with argument unpacking:
         collection.find(**query_builder.build())
        """
        query_spec = {
            "filter": self.base_query,
            "projection": self.projection_fields if self.projection_fields else None,
        }
        if self.sort_fields:
            query_spec["sort"] = self.sort_fields
        if self.skip_count:
            query_spec["skip"] = self.skip_count
        if self.limit_count:
            query_spec["limit"] = self.limit_count

        return query_spec

    def _clone(self) -> "QueryBuilder":
        """
        Internal utility to clone the QueryBuilder, preserving state.
        """
        return QueryBuilder(
            base_query=self.base_query.copy(),
            projection=self.projection_fields.copy() if self.projection_fields else None,
            sort=self.sort_fields.copy(),
            skip_count=self.skip_count,
            limit_count=self.limit_count
        )


class DSL:
    """
    High-level class that ties everything together with a specific PyMongo Collection.
    """

    def __init__(self, collection: Collection):
        self.collection = collection
        self._qb = QueryBuilder()

    def find(self, condition: Condition) -> "DSL":
        new_dsl = DSL(self.collection)
        new_dsl._qb = self._qb.find(condition)
        return new_dsl

    def projection(self, fields: Union[List[str], Dict[str, int]]) -> "DSL":
        new_dsl = DSL(self.collection)
        new_dsl._qb = self._qb.projection(fields)
        return new_dsl

    def sort(self, field: str, ascending: bool = False, descending: bool = False) -> "DSL":
        new_dsl = DSL(self.collection)
        new_dsl._qb = self._qb.sort(field, ascending=ascending, descending=descending)
        return new_dsl

    def skip(self, count: int) -> "DSL":
        new_dsl = DSL(self.collection)
        new_dsl._qb = self._qb.skip(count)
        return new_dsl

    def limit(self, count: int) -> "DSL":
        new_dsl = DSL(self.collection)
        new_dsl._qb = self._qb.limit(count)
        return new_dsl

    def execute(self):
        """
        Execute a find() query with the current query builder state.
        Returns a PyMongo Cursor.
        """
        params = self._qb.build()
        return self.collection.find(**params)

    def count(self) -> int:
        """
        Example of how you might call count_documents() using the current filter.
        """
        return self.collection.count_documents(self._qb.base_query)

    def delete_many(self):
        """
        Example of how you might delete documents matching the current filter.
        """
        return self.collection.delete_many(self._qb.base_query)

    def update_many(self, update_doc: Dict[str, Any]):
        """
        Example of how you might update documents matching the current filter.
        """
        return self.collection.update_many(self._qb.base_query, update_doc)

    def build_query(self) -> Dict[str, Any]:
        """
        Get the final dictionary representation of the query (for debugging or custom usage).
        """
        return self._qb.build()


#
# USAGE EXAMPLE
#
if __name__ == "__main__":
    # Example usage (requires a running MongoDB instance and a collection reference).
    import pymongo

    # Connect to MongoDB (Change the URI & DB name accordingly)
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["testdb"]
    users_collection = db["users"]

    dsl = DSL(users_collection)
    qb = QueryBuilder()
    qb = qb.find(Q("age").gte(21))
    q = qb.build()
    print(q)
    # Build a query equivalent to:
    #    ((age >= 21 AND city = "London") OR role = "admin")
    #    AND status EXISTS
    #    AND tags array has length 3
    # Only fetch `name`, `city` fields, skip 5, limit 10, sorted by `age` descending.
    query = (
            (
                    (Q("age").gte(21) & Q("city").eq("London"))
                    | Q("role").eq("admin")
            )
            & Q("status").exists(True)
            & Q("tags").size(3)
    )

    # Build and execute
    results = (
        dsl.find(query)
        .projection(["name", "city"])
        .sort("age", descending=True)
        .skip(5)
        .limit(10)
        .execute()
    )

    for doc in results:
        print(doc)

    # Example: Counting documents
    count_result = dsl.find(Q("role").eq("admin")).count()
    print("Number of admin users:", count_result)

    # Example: Deleting documents
    # delete_result = dsl.find(Q("obsolete").eq(True)).delete_many()
    # print("Deleted count:", delete_result.deleted_count)

    # Example: Updating documents
    # update_result = dsl.find(Q("role").eq("guest")).update_many({"$set": {"role": "user"}})
    # print("Modified count:", update_result.modified_count)
