import threading
import os

from bson import ObjectId
from loguru import logger
from pymongo import ReplaceOne, DeleteMany
from pymongo.collection import Collection
from tqdm import tqdm
from diskcache import Index

from SLM.appGlue.DesignPaterns.MessageSystem import MessageSystem
from SLM.appGlue.core import  Event
from SLM.iterable.bach_builder import BatchBuilder
import weakref
from collections import OrderedDict


class LRUCache:
    """Простая LRU-реализация на OrderedDict"""

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        """Получить объект из кеша (и переместить в начало как недавно использованный)"""
        if key in self.cache:
            self.cache.move_to_end(key, last=False)  # Переместить в начало
            return self.cache[key]
        return None

    def set(self, key, value):
        """Добавить объект в кеш и удалить самый старый, если кеш переполнен"""
        self.cache[key] = value
        self.cache.move_to_end(key, last=False)  # Переместить в начало
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=True)  # Удалить самый старый элемент


class FieldPropInfo:
    def __init__(self, name, field_type, default):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.converter = None  # todo implement conversion
        self.validator = None  # todo implement validation

    def __get__(self, instance, owner):
        return instance.get_field_val(self.name)

    def __set__(self, instance, value):
        instance.set_field_val(self.name, value)


get_or_create_semaphore = False


class DbRecordMeta(type):
    """Метакласс с LRU-кешем и WeakValueDictionary"""
    _cache = weakref.WeakValueDictionary()  # Храним только живые ссылки
    _lru_cache = LRUCache(max_size=100)  # LRU-хранилище в памяти
    _lock = threading.RLock()

    def __new__(cls, name, bases, namespace, **kwargs):
        print(f"Creating class {name} with kwargs: {kwargs}")
        new_class = super().__new__(cls, name, bases, namespace)
        # do something with kwargs
        new_class.metadata = kwargs.get('table', None)
        print("class created with metadata:", new_class.metadata)
        return new_class

    def __call__(cls, record_id):
        with cls._lock:
            record_id = str(record_id)
            if record_id == "None":
                instance = super().__call__(None)
                id = str(instance._id)
                cls._store_in_cache(id, instance)
                return instance

            if record_id in cls._cache:
                return cls._cache[record_id]

            instance = cls._lru_cache.get(record_id)
            if instance:
                cls._cache[record_id] = instance
                return instance

            instance = super().__call__(record_id)
            cls._store_in_cache(record_id, instance)
            return instance

    @classmethod
    def _store_in_cache(cls, record_id, instance):
        cls._cache[record_id] = instance
        cls._lru_cache.set(record_id, instance)


class MongoRecordWrapper(metaclass=DbRecordMeta):
    client = None
    table_name = None
    onDeleteGlobal: Event = Event()
    verbose = True

    # Инициализация кэша для операций будет выполняться при первом доступе
    attended_bulk_ops = None
    _cache_initialized = False

    @classmethod
    def _ensure_cache_initialized(cls):
        """Убеждаемся, что кэш операций инициализирован"""
        if not cls._cache_initialized:
            cache_dir = os.path.join(os.path.expanduser('~'), '.slm_cache', 'bulk_ops')
            os.makedirs(cache_dir, exist_ok=True)
            cls.attended_bulk_ops = Index(cache_dir)
            cls._cache_initialized = True

    @classmethod
    def add_Delete_many_bulk(cls, query):
        cls._ensure_cache_initialized()
        key = f"DeleteMany_{str(cls.collection().name)}"
        ops_list = cls.attended_bulk_ops.get(key, [])
        ops_list.append(DeleteMany(query))
        cls.attended_bulk_ops[key] = ops_list
        if cls.verbose:
            logger.info(f"Added DeleteMany operation for {cls.collection().name} with query: {query}")

    @staticmethod
    def attended_process_bulk_ops():
        """
        Process all attended bulk operations
        """
        if not MongoRecordWrapper._cache_initialized or not MongoRecordWrapper.attended_bulk_ops:
            return

        # Получаем все ключи из индекса
        all_keys = list(MongoRecordWrapper.attended_bulk_ops.keys())

        for key in all_keys:
            if key.startswith("DeleteMany_"):
                collection_name = key.split("_", 1)[1]
                collection = MongoRecordWrapper.client.db[collection_name]
                bulk_ops = MongoRecordWrapper.attended_bulk_ops.get(key, [])

                if bulk_ops:
                    bach_b = BatchBuilder(bulk_ops, 128)
                    for bach in tqdm(bach_b.bach, desc=f"Processing DeleteMany operations on {collection_name}"):
                        try:
                            collection.bulk_write(bach)
                        except Exception as e:
                            logger.error(f"Error processing batch operation DeleteMany on {collection_name}: {e}")

                    # Удаляем обработанные операции из кэша
                    del MongoRecordWrapper.attended_bulk_ops[key]

        # Очистка не требуется, так как мы удаляем ключи по мере обработки

    def __init__(self, oid):
        if oid is None:
            oid = ObjectId()
        self._id = ObjectId(oid)
        self.props_cache = {}
        self.onDelete: Event = Event()
        self.onEdit: Event = Event()

    @classmethod
    def create_index(cls, index_name, index_fields):
        collection = cls.collection()
        indexes = collection.index_information()
        if index_name in indexes:
            return
        try:
            collection.create_index(index_fields, name=index_name)
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")

    @classmethod
    def collection(cls) -> Collection:

        return MongoRecordWrapper.client.coll(cls)

    def invalidate_cache(self):
        self.props_cache = {}

    @classmethod
    def get_by_id(cls, record_id):
        """
        Get record by id
        :param record_id: id of record
        :return: MongoRecordWrapper instance or None if not found
        """
        if isinstance(record_id, str):
            record_id = ObjectId(record_id)
        res = cls.collection().find_one({"_id": record_id})
        if res is None:
            return None
        return cls(res["_id"])

    @classmethod
    def find_one(cls, query):
        res = cls.collection().find_one(query)
        if res is None: return None
        return cls(res["_id"])

    @classmethod
    def find(cls, query, sort_query=None):
        if sort_query is None:
            res = cls.collection().find(query)
        else:
            res = cls.collection().find(query).sort(sort_query)
        return [cls(item["_id"]) for item in res]

    @classmethod
    def new_record(cls, **kwargs):
        """
        Create new record in db
        for set filed use kwarg name = value as (field_name = value)
        :param kwargs:
        :return:
        """
        return cls._create_new(**kwargs)

    @classmethod
    def _create_new(cls, **kwargs):
        data = cls.create_record_data(**kwargs)
        ins_res = cls.collection().insert_one(data)
        return cls(ins_res.inserted_id)

    def clear_cache(self):
        self.props_cache = {}

    @classmethod
    def _get_or_create(cls, **kwargs):
        data = cls._create_search_data(**kwargs)
        res = cls.find_one(data)
        if res is None:
            new_inst = cls._create_new(**kwargs)
            return new_inst
        else:
            return res

    @classmethod
    def get_or_create(cls, **kwargs):
        return cls._get_or_create(**kwargs)

    @classmethod
    def create_record_data(cls, **kwargs):
        record = {}
        for key, value in cls.__dict__.items():
            if isinstance(value, FieldPropInfo):
                val = kwargs.get(key, value.default)
                record[value.name] = val
        return record

    @classmethod
    def _create_search_data(cls, **kwargs):
        record = {}
        for item in kwargs:
            prop_info = cls.__dict__.get(item, None)
            if prop_info is None:
                Exception()
            search_field_name = prop_info.name
            record[search_field_name] = kwargs[item]
        return record

    def get_field_val(self, field, defaulth=None,hashed=True):
        if field in self.props_cache and hashed:
            return self.props_cache[field]
        self.get_record_data()
        return self.props_cache.get(field, defaulth)

    def set_field_val(self, field, val):
        data = {field: val}
        self.set_record_data(data)

    def delete_rec(self):
        """
        Delete record from db
        :return:
        """
        parent = self
        self.onDelete.fire(parent)
        self.onDeleteGlobal.fire(parent)
        MessageSystem.SendMessage("collection_record_deleted", self._id)
        self.collection().delete_one({"_id": self._id})

    def get_record_data(self):
        res = self.collection().find_one({'_id': self._id})
        if res is None:
            return {}
        self.props_cache = res
        return res

    def set_record_data(self, data):
        query = {'_id': self._id}
        self.collection().update_one(query, {'$set': data}, upsert=True)
        self.get_record_data()

    def list_get(self, name):
        if name in self.props_cache:
            return self.props_cache.get(name, [])
        data = self.get_record_data()
        list = data.get(name, [])
        return list

    def list_extend(self, name, value, no_dupes=False):
        dlist = self.list_get(name)
        dlist.extend(value)
        if no_dupes:
            dlist = list(set(dlist))
        self.set_record_data({name: dlist})

    def list_append(self, name, value, no_dupes=False):
        dlist = self.list_get(name)
        dlist.append(value)
        if no_dupes:
            dlist = list(set(dlist))
        self.set_record_data({name: dlist})

    def list_remove(self, name, val):
        _list = set(self.list_get(name))
        if val in _list:
            _list.remove(val)
            self.set_record_data({name: list(_list)})

    def none_or(self, cond, val):
        if cond is None:
            return None
        return val

    @classmethod
    def update_bulk(cls, file_list: list, upsert=True):
        if len(file_list) == 0 or file_list is None:
            logger.warning('No files to update')
            return
        bulk_op = []
        for record in file_list:
            if record is None:
                continue
            query = {'_id': record['_id']}
            old_data = cls.collection().find_one(query)
            if old_data is not None:
                old_data.update(record)
            bulk_op.append(ReplaceOne(query, old_data, upsert=upsert))
        cls.collection().bulk_write(bulk_op)

    def __hash__(self):
        return hash(str(self._id))

    @property
    def id(self):
        return self._id
