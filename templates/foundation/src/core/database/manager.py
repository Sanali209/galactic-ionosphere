from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
import asyncio

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
        return cls._instance

    async def connect(self, host: str, port: int, db_name: str):
        try:
            conn_str = f"mongodb://{host}:{port}"
            self.client = AsyncIOMotorClient(conn_str)
            self.db = self.client[db_name]
            # Verify connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {conn_str}, DB: {db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_collection(self, name: str):
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[name]

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

# Global instance for ORM access
db_manager = DatabaseManager()
