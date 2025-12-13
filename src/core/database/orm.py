import abc
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from bson import ObjectId
from loguru import logger
from src.core.events import ObserverEvent
from src.core.database.manager import db_manager

T = TypeVar('T', bound='CollectionRecord')

class FieldPropInfo:
    """
    Descriptor for ORM fields.
    Handles validation, conversion, and default values.
    """
    def __init__(self, name: str, default: Any = None, field_type: Type = None, 
                 validator=None, converter=None):
        self.name = name
        self.default = default
        self.field_type = field_type
        self.validator = validator
        self.converter = converter

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.get_field_val(self.name, self.default)

    def __set__(self, instance, value):
        if self.converter:
            value = self.converter(value)
        if self.field_type and not isinstance(value, self.field_type) and value is not None:
             # Basic type check only if value is not None
             # In a real app we might want strict validation or casting
             pass
        if self.validator:
            if not self.validator(value):
                # We could raise an error or just log it
                logger.warning(f"Validation failed for field {self.name} with value {value}")
        
        instance.set_field_val(self.name, value)

class DbRecordMeta(type):
    """Metaclass to handle registration of subclasses for polymorphism."""
    _registry: Dict[str, Type['CollectionRecord']] = {}
    
    def __new__(cls, name, bases, namespace, **kwargs):
        new_class = super().__new__(cls, name, bases, namespace)
        
        # Register class for polymorphism if it's a concrete CollectionRecord subclass
        # We assume immediate subclasses of CollectionRecord might be abstract or concrete
        # Ideally we'd check if 'collection_name' is defined
        table = kwargs.get('table', None)
        if table:
             new_class._collection_name = table
        
        indexes = kwargs.get('indexes', [])
        if indexes:
            new_class._indexes = indexes
        
        # Use the class name as the registry key for _cls
        cls._registry[name] = new_class
        return new_class

class CollectionRecord(metaclass=DbRecordMeta):
    _collection_name: str = None
    _indexes: List[Union[str, List[tuple]]] = []
    
    def __init__(self, oid: Union[str, ObjectId] = None, **kwargs):
        if oid is None:
            oid = ObjectId()
        self._id = ObjectId(oid)
        
        # Initialize events
        # Emits (self, field_name, new_value)
        self.on_change = ObserverEvent(f"Change-{self._id}")
        self.on_delete = ObserverEvent(f"Delete-{self._id}")
        
        # Internal cache
        self._data_cache: Dict[str, Any] = {}
        
        # Populate initial data if provided (e.g. from DB load)
        if kwargs:
             self._data_cache.update(kwargs)

    @property
    def id(self):
        return self._id

    # --- Field Access ---
    
    def get_field_val(self, name: str, default: Any = None):
        return self._data_cache.get(name, default)

    def set_field_val(self, name: str, value: Any):
        old_value = self._data_cache.get(name)
        if old_value != value:
            self._data_cache[name] = value
            # Reactive event
            self.on_change.emit(self, name, value)

    # --- Database Operations (Async) ---
    @classmethod
    def get_collection(cls):
        if not cls._collection_name:
             raise ValueError(f"Class {cls.__name__} must define _collection_name or pass table='name' in metaclass kwargs")
        return db_manager.get_collection(cls._collection_name)

    @classmethod
    async def ensure_indexes(cls):
        """Creates indexes defined in the class."""
        if not cls._indexes:
            return
        coll = cls.get_collection()
        for idx in cls._indexes:
            # idx can be "fieldname" or [("field", 1), ("other", -1)]
            if isinstance(idx, str):
                await coll.create_index(idx)
            else:
                await coll.create_index(idx)

    @classmethod
    async def get(cls: Type[T], oid: Union[str, ObjectId]) -> Optional[T]:
        if isinstance(oid, str):
            oid = ObjectId(oid)
        
        coll = cls.get_collection()
        data = await coll.find_one({"_id": oid})
        if not data:
            return None
            
        return cls._instantiate_from_data(data)

    @classmethod
    async def find(cls: Type[T], query: Dict) -> List[T]:
        coll = cls.get_collection()
        cursor = coll.find(query)
        results = []
        async for doc in cursor:
            results.append(cls._instantiate_from_data(doc))
        return results

    @classmethod
    async def find_one(cls: Type[T], query: Dict) -> Optional[T]:
        coll = cls.get_collection()
        data = await coll.find_one(query)
        if data:
            return cls._instantiate_from_data(data)
        return None

    @classmethod
    def _instantiate_from_data(cls, data: Dict) -> 'CollectionRecord':
        # Polymorphism support: check if there's a specific class defined
        cls_name = data.get('_cls')
        target_cls = cls
        if cls_name and cls_name in DbRecordMeta._registry:
            target_cls = DbRecordMeta._registry[cls_name]
        
        # Create instance bypassing __init__ or using it carefully
        # Here we assume __init__ handles kwargs to populate cache
        obj = target_cls(oid=data['_id'], **data)
        return obj

    @classmethod
    async def aggregate(cls: Type[T], pipeline: List[Dict], as_model: bool = False) -> List[Union[Dict, T]]:
        """
        Execute an aggregation pipeline.
        
        :param pipeline: List of aggregation stages.
        :param as_model: If True, attempts to convert results to CollectionRecord instances.
        :return: List of dicts or CollectionRecord instances.
        """
        coll = cls.get_collection()
        final_pipeline = pipeline.copy()
        
        # Polymorphism Safety: If this class is a subclass (has a specific name registry entry 
        # that implies it is not the base if base uses same collection), filter by _cls.
        # Simple check: if this class name is in registry and we are sharing collection?
        # A safer bet for Single Table Inheritance is: if cls is not the base collection owner 
        # or if we just want to be safe, always filter by _cls if _cls is expected.
        # However, the base class usually doesn't have _cls filter if it wants all.
        
        # Check if we need to filter by _cls
        # If we are strictly querying a subclass, we should filter.
        # We can know this if we find ourselves in the registry and we are not the base?
        # For simplicity in this v1: if the class has subclasses or is a subclass, 
        # we might want to be explicit.
        # But let's assume the user knows what they are doing OR enforce strictness.
        # Strategy: If cls has a concrete _cls name that it saves as, filter by it.
        
        # Assumption: The class name IS the _cls value.
        target_cls_name = cls.__name__
        
        # We only filter if we are NOT the base class defining the collection, 
        # OR if we want to restrict to exactly this type. 
        # Usually in STI, `User.objects` returns all, `Admin.objects` returns Admins.
        # We need a way to detect if we are the "Root" of the collection.
        # Current DbRecordMeta doesn't explicitly mark "Root".
        # Heuristic: If we are a registered subclass, we inject the match.
        # But `User` is also registered.
        
        # Simplified Polymorphism Logic for Aggregate:
        # If the user manually adds a $match on _cls, we trust them.
        # If not, and we are a subclass (how to tell?), we inject.
        # Let's verify via the _cls field existence in a saved record? No.
        
        # For now, let's inject ONLY if `_cls` is not in the first stage match
        # AND if we are likely a subclass.
        # Better approach: The user is calling Admin.aggregate(...) -> likely needs Admin only.
        # User.aggregate(...) -> likely needs all.
        
        # Let's just append the match if the user is asking for specific class hydration
        # OR simply document that aggregate is raw power.
        # BUT, the goal was "Best Practices". 
        
        # Implementation: Only inject if `as_model=True`? No, data analysis on subclass should be scoped.
        
        # Let's rely on `_registry`. If `cls` is in registry, we COULD filter.
        # But `User` is in registry.
        # Let's skip auto-filtering for now to allow full flexibility unless we add a specific flag.
        # actually, the requirement in plan was: "Automatically inject ... if the class is a subclass".
        # Since I generally lack the metadata "is_subclass", I will add a manual check:
        # If `cls` in registry, and `cls` is NOT the base (we can't easily check base without finding parent).
        pass 
        
        # Re-reading plan: "Automatically inject a $match: {"_cls": "ClassName"}"
        # Let's do it for ALL classes for now, except if it's the base? 
        # Actually proper STI usually filters `_cls` for all queries on that model.
        # But if `User` is base, `User.aggregate` should see `Admin` too.
        # So we only filter if we are a Derived class.
        # We can check `cls.__bases__`.
        
        is_subclass = False
        for base in cls.__bases__:
            if issubclass(base, CollectionRecord) and base is not CollectionRecord:
                 # It inherits from another CollectionRecord, likely a subclass
                 is_subclass = True
                 break
        
        if is_subclass:
             # It's a derived class (e.g. Admin(User)), inject filter
             final_pipeline.insert(0, {"$match": {"_cls": cls.__name__}})

        cursor = coll.aggregate(final_pipeline)
        results = []
        async for doc in cursor:
            if as_model:
                # Attempt to convert. Doc must have _id.
                if '_id' in doc:
                    results.append(cls._instantiate_from_data(doc))
                else:
                    # Fallback or strict error? 
                    # If projection removes _id, we can't fully hydrate.
                    # Best effort: return dict or partial object if possible?
                    # Let's return dict if hydration fails due to missing keys?
                    # Or just wrap what we have. `_instantiate_from_data` needs `_id`.
                    results.append(doc)
            else:
                results.append(doc)
        return results

    async def save(self):
        coll = self.get_collection()
        data = self._data_cache.copy()
        data['_id'] = self._id
        data['_cls'] = self.__class__.__name__ # Store class name for polymorphism
        
        await coll.replace_one({'_id': self._id}, data, upsert=True)
        # We could also do partial updates with $set for efficiency

    @classmethod
    async def aggregate(cls: Type[T], pipeline: List[Dict], as_model: bool = False) -> List[Union[Dict, T]]:
        """
        Execute an aggregation pipeline.
        
        :param pipeline: List of aggregation stages.
        :param as_model: If True, attempts to convert results to CollectionRecord instances.
        :return: List of dicts or CollectionRecord instances.
        """
        coll = cls.get_collection()
        final_pipeline = pipeline.copy()
        
        # Polymorphism Safety
        is_subclass = False
        for base in cls.__bases__:
            if issubclass(base, CollectionRecord) and base is not CollectionRecord:
                 is_subclass = True
                 break
        
        if is_subclass:
             final_pipeline.insert(0, {"$match": {"_cls": cls.__name__}})

        cursor = coll.aggregate(final_pipeline)
        results = []
        async for doc in cursor:
            if as_model and '_id' in doc:
                 results.append(cls._instantiate_from_data(doc))
            else:
                results.append(doc)
        return results

    async def delete(self):
        coll = self.get_collection()
        await coll.delete_one({'_id': self._id})
        self.on_delete.emit(self)

    # --- List/Dict Helpers ---
    # These would need to handle mutable types carefully.
    # If a user appends to a list stored in a field, the descriptor __set__ isn't called.
    # A standard workaround is to provide specific methods for mutating these fields.

    def list_append(self, field_name: str, item: Any):
        current_list = self.get_field_val(field_name, [])
        if not isinstance(current_list, list):
             current_list = []
        current_list.append(item)
        self.set_field_val(field_name, current_list)
    
    def dict_update(self, field_name: str, key: str, value: Any):
        current_dict = self.get_field_val(field_name, {})
        if not isinstance(current_dict, dict):
             current_dict = {}
        current_dict[key] = value
        self.set_field_val(field_name, current_dict)

