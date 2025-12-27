import os
from pymongo import MongoClient
from pymongo.database import Database
from loguru import logger
from typing import Optional, Dict, Type, List
from collections import defaultdict

from SLM.core.component import Component
from SLM.core.message_bus import MessageBus
from .queryset import QuerySet
from . import DOCUMENT_REGISTRY

class MongoODMComponent(Component):
    """
    A unified component that manages MongoDB connection, document registration,
    object manager injection, and index creation.
    """

    def __init__(self, config, message_bus: Optional[MessageBus] = None):
        self.config = config
        self.message_bus = message_bus
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._registered_documents: Dict[str, List[Type]] = defaultdict(list)

    def start(self):
        """Connects to the database, registers documents, injects managers, and creates indexes."""
        db_config = self.config.get("mongodb")
        if not db_config:
            raise ValueError("MongoDB configuration is missing.")

        try:
            self.client = MongoClient(
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 27017),
            )
            # Ping the server to verify connection, but skip for mock clients
            if "mongomock" not in str(type(self.client)):
                self.client.admin.command('ping')
            self.db = self.client[db_config["db_name"]]
            logger.info(f"Successfully connected to MongoDB at {db_config.get('host', 'localhost')}:{db_config.get('port', 27017)}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

        self._register_documents_from_registry()
        self._inject_object_managers()
        self._create_all_indexes()
        logger.info("MongoODMComponent started, object managers injected, and indexes created.")

    def stop(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def get_database(self) -> Database:
        """Returns the database instance."""
        if self.db is None:
            raise Exception("Database not initialized. Is the component running?")
        return self.db

    def get_collection(self, collection_name: str):
        """Get a collection from the database."""
        return self.get_database()[collection_name]

    def _register_documents_from_registry(self):
        """Registers all Document classes found in the global registry."""
        for cls in DOCUMENT_REGISTRY:
            if not hasattr(cls, '__collection__'):
                raise TypeError(f"Document class {cls.__name__} must have a '__collection__' attribute.")
            
            collection_name = cls.__collection__
            if cls not in self._registered_documents[collection_name]:
                self._registered_documents[collection_name].append(cls)
                logger.info(f"Registered document {cls.__name__} for collection '{collection_name}'.")

    def _inject_object_managers(self):
        """Injects the QuerySet 'objects' manager into each registered Document class."""
        for collection_name, doc_classes in self._registered_documents.items():
            collection = self.get_collection(collection_name)
            base_doc_class = doc_classes[0] 
            
            queryset_instance = QuerySet(
                document_class=base_doc_class, 
                collection=collection,
                message_bus=self.message_bus
            )
            
            for doc_class in doc_classes:
                doc_class.objects = queryset_instance
                logger.info(f"Injected 'objects' manager into {doc_class.__name__} for collection '{collection_name}'.")

            for doc_class in doc_classes:
                for base in doc_class.__mro__:
                    if base != doc_class and hasattr(base, '__abstract__') and base.__abstract__:
                        if not hasattr(base, 'objects') or base.objects is None:
                            base.objects = queryset_instance
                            logger.info(f"Injected 'objects' manager into abstract base {base.__name__} for collection '{collection_name}'.")

    def _create_all_indexes(self):
        """Iterates through all registered documents and creates their defined indexes."""
        for doc_classes in self._registered_documents.values():
            for doc_class in doc_classes:
                self._create_indexes_for_class(doc_class)

    def _create_indexes_for_class(self, doc_class):
        """Creates the indexes for a single document class."""
        if not hasattr(doc_class, '_indexes') or not doc_class._indexes:
            return

        collection_name = doc_class.__collection__
        collection = self.get_collection(collection_name)
        
        try:
            for index_definition in doc_class._indexes:
                if isinstance(index_definition, list):
                    # Simple index: [('field1', 1), ('field2', -1)]
                    collection.create_index(index_definition)
                    logger.info(f"Created index on {index_definition} for collection '{collection_name}'.")
                elif isinstance(index_definition, dict):
                    # Complex index: {'fields': [...], 'name': 'my_index', ...}
                    fields = index_definition.pop('fields')
                    collection.create_index(fields, **index_definition)
                    logger.info(f"Created index '{index_definition.get('name')}' on {fields} for collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Error creating indexes for {doc_class.__name__} on collection '{collection_name}': {e}")
