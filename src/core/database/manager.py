from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from loguru import logger

class MongoManager:
    def __init__(self):
        self.client: AsyncMongoClient = None
        self.db: AsyncDatabase = None
    
    def init(self):
        import asyncio
        from src.core.locator import sl
        try:
             loop = asyncio.get_running_loop()
             logger.info(f"DEBUG: MongoManager init on loop: {loop}")
        except Exception as e:
             logger.error(f"DEBUG: MongoManager init NO RUNNING LOOP: {e}")

        config = sl.config.data.mongo
        connection_url = f"mongodb://{config.host}:{config.port}"
        try:
            self.client = AsyncMongoClient(connection_url)
            self.db = self.client[config.database_name]
            logger.info(f"Connected to MongoDB (Async): {connection_url}/{config.database_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

    async def reset_db(self):
        """Drops the entire database."""
        if self.client is not None and self.db is not None:
            name = self.db.name
            await self.client.drop_database(name)
            logger.warning(f"Database '{name}' dropped.")
            # Re-init? The db object might be invalid.
            # Usually PyMongo handles this, but let's be safe.
            self.db = self.client[name]

    def get_collection(self, collection_name: str):
        if self.db is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self.db[collection_name]

# Global instance for easy access, similar to ServiceLocator pattern but specific to DB
# Alternatively, could be accessed via sl.caps.get_driver("storage") etc.
db_manager = MongoManager()
