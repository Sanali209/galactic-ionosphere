from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from ..base_system import BaseSystem

class DatabaseManager(BaseSystem):
    """
    Manages MongoDB connection using the foundation BaseSystem pattern.
    """
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self.client = None
        self.db = None
        
        # Backward compatibility for ORM
        global db_manager
        db_manager = self

    async def initialize(self):
        """Connect to database using config."""
        try:
            # Get config or use defaults
            # ConfigManager loads into .data which is AppConfig pydantic model
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'mongo'):
                mongo_cfg = self.config.data.mongo
                host = mongo_cfg.host
                port = mongo_cfg.port
                db_name = mongo_cfg.database_name
            else:
                # Fallback if config not loaded properly
                host = 'localhost'
                port = 27017
                db_name = 'app_db'
            
            conn_str = f"mongodb://{host}:{port}"
            self.client = AsyncIOMotorClient(conn_str)
            self.db = self.client[db_name]
            
            # Verify connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {conn_str}, DB: {db_name}")
            
            await super().initialize()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def shutdown(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")
        await super().shutdown()

    def get_collection(self, name: str):
        if self.db is None:
            raise RuntimeError("Database not connected. Call initialize() first.")
        return self.db[name]

# Global instance for ORM access (will be set during init)
db_manager = None
