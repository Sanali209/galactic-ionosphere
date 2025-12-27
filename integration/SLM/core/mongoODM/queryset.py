from pymongo.collection import Collection
from SLM.core.message_bus import MessageBus
from typing import Optional
from SLM.core.mongoODM import DOCUMENT_REGISTRY

class QuerySet:
    """
    Provides an interface for querying a collection for a specific Document type.
    """
    def __init__(self, document_class, collection: Collection, message_bus: Optional[MessageBus] = None):
        self._document_class = document_class
        self._collection = collection
        self._message_bus = message_bus
        self.polymorphic_map = self._build_poly_map()

    def _build_poly_map(self):
        """Builds a map of class names to class objects for this collection."""
        poly_map = {}
        collection_name = self._collection.name
        for doc_class in DOCUMENT_REGISTRY:
            if doc_class.__collection__ == collection_name:
                poly_map[doc_class.__name__] = doc_class
        return poly_map

    def _from_mongo(self, data):
        """
        Helper to convert raw mongo data to a document instance, handling polymorphism.
        """
        if not data:
            return None

        cls_name = data.get('_cls')
        doc_class = self.polymorphic_map.get(cls_name, self._document_class)
        
        doc = doc_class.from_mongo(data)
        if doc:
            doc._set_queryset(self)
        return doc

    def find(self, filter=None, sort=None, **kwargs):
        """
        Find multiple documents and return a cursor of Document instances.
        """
        cursor = self._collection.find(filter, **kwargs)
        if sort:
            cursor = cursor.sort(sort)
        for data in cursor:
            yield self._from_mongo(data)

    def find_one(self, filter=None, **kwargs):
        """
        Find a single document and return a Document instance.
        """
        data = self._collection.find_one(filter, **kwargs)
        return self._from_mongo(data)

    def insert_one(self, data):
        """
        Insert a single document.
        """
        return self._collection.insert_one(data)

    def update_one(self, filter, update, **kwargs):
        """
        Update a single document.
        """
        return self._collection.update_one(filter, update, **kwargs)

    def delete_one(self, filter, **kwargs):
        """
        Delete a single document.
        """
        return self._collection.delete_one(filter, **kwargs)

    def count_documents(self, filter=None, **kwargs) -> int:
        """
        Count the number of documents matching a filter.
        """
        if filter is None:
            filter = {}
        return self._collection.count_documents(filter, **kwargs)
