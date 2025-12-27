from __future__ import annotations
import weakref
import threading
from typing import Optional
from .fields import Field
from . import DOCUMENT_REGISTRY
from .queryset import QuerySet

class ODMBase(type):
    _fields: dict
    _indexes: list
    objects: Optional[QuerySet]
    _cache: weakref.WeakValueDictionary
    _lock: threading.RLock

    """
    Metaclass for all document types. It's responsible for:
    1. Collecting all fields defined on the document class.
    2. Setting the `_fields` attribute on the class.
    3. Setting the `db_field` for each field if not provided.
    4. Implementing an identity map pattern using a weak reference cache.
    """

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        # Initialize cache and lock for each class
        new_class._cache = weakref.WeakValueDictionary()
        new_class._lock = threading.RLock()

        # Collect fields from the class and its bases
        fields = {}
        for base in reversed(new_class.__mro__):
            if hasattr(base, '_fields'):
                fields.update(base._fields)
        
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                fields[attr_name] = attr_value
                attr_value.name = attr_name
                if attr_value.db_field is None:
                    attr_value.db_field = attr_name

        new_class._fields = fields
        
        # Collect index information from fields
        indexes = []
        for name, field in fields.items():
            if field.unique:
                indexes.append({'fields': [(field.db_field or name, 1)], 'unique': True})
            elif field.index:
                indexes.append({'fields': [(field.db_field or name, 1)]})

        # Collect index information from Meta class
        if 'Meta' in attrs:
            meta_attrs = attrs['Meta'].__dict__
            if 'indexes' in meta_attrs:
                indexes.extend(meta_attrs['indexes'])
        
        new_class._indexes = indexes

        # The 'objects' manager will be attached later, once the DB components are ready
        is_document_subclass = any('Document' in [b.__name__ for b in base.__mro__] for base in bases)
        if is_document_subclass and not attrs.get('__abstract__'):
             new_class.objects = None # Placeholder for the QuerySet manager
             if new_class not in DOCUMENT_REGISTRY:
                DOCUMENT_REGISTRY.append(new_class)

        return new_class

    def __call__(cls, *args, **kwargs):
        """
        Override the default class instantiation process to implement
        an identity map. When a document is instantiated, we check if an
        instance with the same primary key already exists in our cache.
        """
        # Embedded documents don't have a pk and are not cached
        if 'EmbeddedDocument' in [b.__name__ for b in cls.__mro__]:
            return super().__call__(*args, **kwargs)

        pk = kwargs.get('pk') or kwargs.get('_id')
        if pk is None:
            # For new documents, just create a new instance
            return super().__call__(*args, **kwargs)

        with cls._lock:
            # Check if an instance with this pk is already cached
            instance = cls._cache.get(pk)
            if instance:
                # If found, re-initialize it with new data if provided
                # This is useful for updates from the database
                instance.__init__(*args, **kwargs)
                return instance
            
            # If not in cache, create a new instance
            instance = super().__call__(*args, **kwargs)
            
            # Store the new instance in the cache
            cls._cache[instance.pk] = instance
            return instance
