from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from loguru import logger
from src.core.locator import sl

class MongoManager:
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db: AsyncIOMotorDatabase = None
    
    def init(self):
        config = sl.config.data.mongo
        connection_url = f"mongodb://{config.host}:{config.port}"
        try:
            self.client = AsyncIOMotorClient(connection_url)
            self.db = self.client[config.database_name]
            logger.info(f"Connected to MongoDB: {connection_url}/{config.database_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

    def get_collection(self, collection_name: str):
        if self.db is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self.db[collection_name]

# Global instance for easy access, similar to ServiceLocator pattern but specific to DB
# Alternatively, could be accessed via sl.caps.get_driver("storage") etc.
db_manager = MongoManager()
