import abc
import asyncio
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic, Sequence, Iterator
from bson import ObjectId
from loguru import logger
from ..events import Signal

T = TypeVar('T', bound='CollectionRecord')

# --- Utility Functions ---

def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.
    
    Examples:
        ImageRecord -> image_record
        HTTPResponse -> http_response
        MyAPIClient -> my_api_client
    """
    # Insert underscore before any uppercase letter that follows a lowercase letter
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before any uppercase letter that follows a lowercase or uppercase letter
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# --- Field Descriptors ---

class Field(abc.ABC):
    """Base class for all ORM fields."""
    def __init__(self, name: str = None, default: Any = None, default_factory: callable = None, index: bool = False, unique: bool = False):
        self.name = name # Set by metaclass
        self.default = default
        self.default_factory = default_factory
        self.index = index
        self.unique = unique

    def __get__(self, instance, owner):
        if instance is None: return self
        # Use factory if provided and no value exists
        if self.default_factory is not None:
            current_val = instance.get_field_val(self.name, None)
            if current_val is None:
                val = self.default_factory()
                instance.set_field_val(self.name, val)
                return val
        return instance.get_field_val(self.name, self.default)

    def validate(self, value: Any) -> Any:
        """Validate the value. Override in subclasses."""
        return value

    def __set__(self, instance, value):
        value = self.validate(value)
        instance.set_field_val(self.name, value)
    
    def to_mongo(self, value: Any) -> Any:
        return value

    def from_mongo(self, value: Any) -> Any:
        return value


class StringField(Field):
    def validate(self, value: Any) -> str:
        if value is None:
            return None
        return str(value)

class IntField(Field):
    def validate(self, value: Any) -> int:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            raise TypeError(f"Field '{self.name}' expects int, got {type(value).__name__}: {value}")


class BoolField(Field):
    def __set__(self, instance, value):
        if value is not None:
            value = bool(value)
        super().__set__(instance, value)

class DictField(Field):
    def __init__(self, **kwargs):
        super().__init__(default=lambda: {}, **kwargs)
        
    def __get__(self, instance, owner):
        val = super().__get__(instance, owner)
        if val is None and callable(self.default):
            val = self.default()
            instance.set_field_val(self.name, val)
        return val

class Reference:
    """Proxy object for lazy loading a related document."""
    def __init__(self, ref_cls: Type['CollectionRecord'], oid: Union[ObjectId, str]):
        self.ref_cls = ref_cls
        self.id = oid if isinstance(oid, ObjectId) else ObjectId(oid)
        self._cache = None

    async def fetch(self) -> Optional['CollectionRecord']:
        if self._cache: return self._cache
        self._cache = await self.ref_cls.get(self.id)
        return self._cache
        
    def __repr__(self):
        return f"<Reference to {self.ref_cls.__name__} id={self.id}>"

class ReferenceList(Sequence):
    """
    Wraps a list of Reference objects to provide convenient batch fetching.
    """
    def __init__(self, references: List[Reference]):
        self._refs = references

    def __getitem__(self, index):
        return self._refs[index]

    def __len__(self):
        return len(self._refs)
        
    def __iter__(self) -> Iterator[Reference]:
        return iter(self._refs)

    async def fetch_all(self) -> List['CollectionRecord']:
        """Fetch all referenced methods concurrently."""
        tasks = [ref.fetch() for ref in self._refs]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    def add(self, item: Union['CollectionRecord', Reference, ObjectId], ref_cls: Type['CollectionRecord'] = None) -> None:
        """
        Append item to the reference list.
        
        Args:
            item: CollectionRecord instance, Reference, or ObjectId to add
            ref_cls: Required when adding ObjectId directly (specifies target model)
            
        Raises:
            ValueError: If ObjectId provided without ref_cls
            TypeError: If item type is not supported
        """
        if isinstance(item, Reference):
            self._refs.append(item)
        elif isinstance(item, CollectionRecord):
            self._refs.append(Reference(type(item), item.id))
        elif isinstance(item, (ObjectId, str)):
            if ref_cls is None:
                raise ValueError("ref_cls required when adding ObjectId/str directly")
            oid = ObjectId(item) if isinstance(item, str) else item
            self._refs.append(Reference(ref_cls, oid))
        else:
            raise TypeError(f"Cannot add {type(item).__name__} to ReferenceList")


class ReferenceField(Field):
    def __init__(self, model_cls: Type['CollectionRecord'], **kwargs):
        super().__init__(**kwargs)
        self.model_cls = model_cls

    def to_mongo(self, value: Any) -> Any:
        if isinstance(value, Reference):
            return value.id
        if isinstance(value, CollectionRecord):
            return value.id
        if isinstance(value, (str, ObjectId)):
            return ObjectId(value)
        return None

    def from_mongo(self, value: Any) -> Any:
        if value is None: return None
        return Reference(self.model_cls, value)

class EmbeddedField(Field):
    def __init__(self, model_cls: Type['CollectionRecord'], **kwargs):
        super().__init__(**kwargs)
        self.model_cls = model_cls

    def to_mongo(self, value: Any) -> Any:
        if value is None: return None
        if isinstance(value, self.model_cls):
            # Embedded docs don't need _id usually, but CollectionRecord has it.
            # We strip _id for embedding if it's strictly structural, 
            # OR we keep it if we want addressable embedded objects.
            # For simplicity, we dump the whole dict.
            data = value.to_dict()
            if '_id' in data: del data['_id'] # usually embedded doesn't need root ID
            return data
        return value

    def from_mongo(self, value: Any) -> Any:
        if value is None: return None
        if isinstance(value, dict):
             # Hydrate
             obj = self.model_cls(**value)
             return obj
        return value

class ListField(Field):
    def __init__(self, inner_field: Field, **kwargs):
        super().__init__(default=lambda: [], **kwargs)
        self.inner_field = inner_field

    def to_mongo(self, value: Any) -> List:
        if not value: return []
        # If it's a ReferenceList, unwrap it
        if isinstance(value, ReferenceList):
             return [self.inner_field.to_mongo(v) for v in value]
        return [self.inner_field.to_mongo(v) for v in value]

    def from_mongo(self, value: List) -> Any:
        if not value: return []
        items = [self.inner_field.from_mongo(v) for v in value]
        
        # If the inner items are References, return a ReferenceList
        if items and isinstance(items[0], Reference):
            return ReferenceList(items)
            
        return items



# --- Metaclass & Record ---

class DbRecordMeta(type):
    """Metaclass to registry models and setup fields."""
    _registry: Dict[str, Type['CollectionRecord']] = {}
    
    def __new__(cls, name, bases, namespace, **kwargs):
        new_class = super().__new__(cls, name, bases, namespace)
        
        # 1. Register
        cls._registry[name] = new_class
        
        # 2. Setup _collection_name
        table = kwargs.get('table', None)
        if table:
             new_class._collection_name = table
        elif '_collection_name' not in namespace or namespace['_collection_name'] is None:
            # Auto-generate collection name from class name
            # Skip for base CollectionRecord class
            if name != 'CollectionRecord':
                # ImageRecord -> image_records
                # SearchHistory -> search_histories  
                snake_name = camel_to_snake(name)
                # Simple pluralization: add 's' (can be enhanced with inflect library)
                if snake_name.endswith('y'):
                    new_class._collection_name = snake_name[:-1] + 'ies'
                elif snake_name.endswith('s'):
                    new_class._collection_name = snake_name + 'es'
                else:
                    new_class._collection_name = snake_name + 's'
        
        # 3. Harvest Fields
        new_class._fields = {}
        for k, v in namespace.items():
            if isinstance(v, Field):
                v.name = k # Inject name
                new_class._fields[k] = v
        
        # 4. Harvest Indices
        new_class._indexes = kwargs.get('indexes', [])
        
        return new_class

class CollectionRecord(metaclass=DbRecordMeta):
    _collection_name: str = None
    _fields: Dict[str, Field] = {}
    _indexes: List = []

    def __init__(self, oid: Union[str, ObjectId] = None, **kwargs):
        if oid is None:
            oid = ObjectId()
        self._id = ObjectId(oid)
        self.on_change = Signal(f"Change-{self._id}")
        self._data_cache: Dict[str, Any] = {}
        
        # Hydrate from kwargs
        if kwargs:
            for k, v in kwargs.items():
                if k == '_id': continue
                if k in self._fields:
                    # If coming from raw DB data (e.g. Reference ID), we might need conversion?
                    # The __init__ is ambiguous: is it raw data or python objects?
                    # Let's assume kwargs are Python objects if passed directly, 
                    # but if using _instantiate_from_data, we process them.
                    self.set_field_val(k, v)
                else:
                    self._data_cache[k] = v

    @property
    def id(self): return self._id

    def get_field_val(self, name: str, default: Any = None):
        return self._data_cache.get(name, default)

    def set_field_val(self, name: str, value: Any):
        old_value = self._data_cache.get(name)
        if old_value != value:
            self._data_cache[name] = value
            self.on_change.emit(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to MongoDB-ready dict."""
        out = {'_id': self._id, '_cls': self.__class__.__name__}
        
        for name, field in self._fields.items():
            val = self.get_field_val(name, field.default)
            if callable(val) and name not in self._data_cache: # handle default factories
                 val = val() 
            out[name] = field.to_mongo(val)
            
        # Include dynamic fields not in schema?
        for k, v in self._data_cache.items():
            if k not in self._fields:
                out[k] = v
        return out

    @classmethod
    def get_collection(cls):
        from src.core.database.manager import DatabaseManager
        db = DatabaseManager.get_instance()
        
        if not cls._collection_name:
             # Try to find from bases
             for base in cls.__bases__:
                 if issubclass(base, CollectionRecord) and getattr(base, '_collection_name', None):
                     return db.get_collection(base._collection_name)
             raise ValueError(f"Class {cls.__name__} must define _collection_name")
        return db.get_collection(cls._collection_name)

    @classmethod
    async def ensure_indexes(cls):
        """Creates indexes defined in Fields and Meta."""
        coll = cls.get_collection()
        
        # 1. Field Indices
        for name, field in cls._fields.items():
            if field.index:
                unique = field.unique
                logger.info(f"Creating index for {cls.__name__}.{name} (unique={unique})")
                await coll.create_index([(name, 1)], unique=unique, background=True)
        
        # 2. Compound/Meta Indices
        for idx in cls._indexes:
            logger.info(f"Creating compound index for {cls.__name__}: {idx}")
            await coll.create_index(idx, background=True)

    @classmethod
    async def get(cls: Type[T], oid: Union[str, ObjectId]) -> Optional[T]:
        if isinstance(oid, str): oid = ObjectId(oid)
        coll = cls.get_collection()
        data = await coll.find_one({"_id": oid})
        return cls._instantiate_from_data(data) if data else None

    @classmethod
    async def find(cls: Type[T], query: Dict, sort: List[tuple] = None, limit: int = None, skip: int = None) -> List[T]:
        """
        Find documents matching query.
        
        Args:
            query: MongoDB query dict
            sort: List of (field, direction) tuples, e.g. [("timestamp", -1)]
            limit: Maximum number of results
            skip: Number of documents to skip (for pagination)
            
        Returns:
            List of model instances
        """
        coll = cls.get_collection()
        cursor = coll.find(query)
        
        # Apply sort if provided
        if sort:
            cursor = cursor.sort(sort)
        
        # Apply skip if provided (for pagination)
        if skip:
            cursor = cursor.skip(skip)
        
        # Apply limit if provided
        if limit:
            cursor = cursor.limit(limit)
        
        results = []
        async for doc in cursor:
            results.append(cls._instantiate_from_data(doc))
        return results

    @classmethod
    async def find_one(cls: Type[T], query: Dict) -> Optional[T]:
        coll = cls.get_collection()
        data = await coll.find_one(query)
        return cls._instantiate_from_data(data) if data else None

    @classmethod
    def _instantiate_from_data(cls, data: Dict) -> 'CollectionRecord':
        if not data: return None
        cls_name = data.get('_cls')
        target_cls = cls
        if cls_name and cls_name in DbRecordMeta._registry:
            target_cls = DbRecordMeta._registry[cls_name]
        
        # Convert fields from Mongo
        processed_kwargs = {}
        for k, v in data.items():
            if k in target_cls._fields:
                field = target_cls._fields[k]
                processed_kwargs[k] = field.from_mongo(v)
            else:
                processed_kwargs[k] = v
        
        return target_cls(oid=data.get('_id'), **processed_kwargs)

    @classmethod
    async def delete_many(cls, query: Dict) -> Any:
        """
        Delete multiple documents matching query.
        
        Args:
            query: MongoDB query dict
            
        Returns:
            DeleteResult from motor/pymongo
        """
        coll = cls.get_collection()
        return await coll.delete_many(query)

    async def save(self, emit_event: bool = True):
        coll = self.get_collection()
        data = self.to_dict()
        await coll.replace_one({'_id': self._id}, data, upsert=True)
        
        if emit_event:
            # Emit event via DatabaseManager bridge
            from src.core.database.manager import DatabaseManager
            try:
                db_manager = DatabaseManager.get_instance()
                collection_name = self.get_collection().name
                
                await db_manager.emit_db_event(
                    f"db.{collection_name}.updated",
                    {
                        "collection": collection_name,
                        "id": self._id,
                        "record": data
                    }
                )
            except Exception:
                # Ignore errors during event emission to prevent save failure
                pass

    async def delete(self, emit_event: bool = True):
        coll = self.get_collection()
        await coll.delete_one({'_id': self._id})
        
        if emit_event:
            from src.core.database.manager import DatabaseManager
            try:
                db_manager = DatabaseManager.get_instance()
                collection_name = self.get_collection().name
                
                await db_manager.emit_db_event(
                    f"db.{collection_name}.deleted",
                    {
                        "collection": collection_name,
                        "id": self._id
                    }
                )
            except Exception:
                pass

    @classmethod
    async def count(cls, query: Dict = None) -> int:
        """
        Count documents matching query.
        
        Args:
            query: MongoDB query dict (optional)
            
        Returns:
            Number of documents matching query
        """
        coll = cls.get_collection()
        if query:
            return await coll.count_documents(query)
        return await coll.estimated_document_count()
