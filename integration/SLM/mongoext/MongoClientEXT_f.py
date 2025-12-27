from loguru import logger
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection

from SLM.appGlue.core import TypedConfigSection, Allocator, Service


class MongoConfig(TypedConfigSection):
    host: str = 'localhost'
    port: int = 27017
    database_name: str = "mongo_wrapper_db"


Allocator.config.register_section("mongoConfig", MongoConfig)


class MongoClientExt(Service):
    ent_map = {
    }
    collections_names = []

    def __init__(self):
        super().__init__()
        self.mongo_db = None
        self.mongo_client = None

    @staticmethod
    def table_from_ent_type(ent_type):
        return MongoClientExt.ent_map.get(ent_type, None)

    @staticmethod
    def add_ent_type_to_map(ent_type, table_name):
        MongoClientExt.ent_map[ent_type] = table_name

    def init(self, config):
        db_config: MongoConfig = config.mongoConfig
        # list of used collections
        self.mongo_client = MongoClient(db_config.host, db_config.port)
        self.mongo_db = self.mongo_client[db_config.database_name]
        print(f"Connected to {db_config.host}:{db_config.port}/{db_config.database_name}")

    def register_collection(self, coll_name, map_key=None):
        if coll_name not in MongoClientExt.collections_names:
            MongoClientExt.collections_names.append(coll_name)
        if map_key is not None:
            MongoClientExt.add_ent_type_to_map(map_key, coll_name)
        self.create_exist_collection(coll_name)

    def check_collection_exist(self, collection_name):
        return collection_name in self.mongo_db.list_collection_names()

    def create_exist_collection(self, coll_name):
        if not self.check_collection_exist(coll_name):
            # create coll
            self.mongo_db.create_collection(coll_name)

    def coll(self, name) -> Collection:
        table_name = MongoClientExt.table_from_ent_type(name)
        if table_name is not None:
            try:
                return self.mongo_db[table_name]
            except Exception as e:
                logger.error(f"Error getting collection {table_name} - {e}")
                return None
        else:
            return self.mongo_db[name]

    def bulk_update(self, records: list, collection_name, upsert=True):
        """
        #todo: refactor this for move to wrapper level
        :param records:
        :param collection_name:
        :param upsert:
        :return:
        """
        if len(records) == 0 or records is None:
            logger.warning('No files to update')
            return
        bulk_op = []
        for record in records:
            if record is None:
                continue
            query = {'_id': record['_id']}
            bulk_op.append(ReplaceOne(query, record, upsert=upsert))
        self.coll(collection_name).bulk_write(bulk_op)
