from typing import TYPE_CHECKING, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from ..base_system import BaseSystem

if TYPE_CHECKING:
    from ..locator import ServiceLocator


class DatabaseManager(BaseSystem):
    """
    Manages MongoDB connection using the foundation BaseSystem pattern.
    
    Access via ServiceLocator:
        db = DatabaseManager.get_instance()
        collection = db.get_collection("my_collection")
    
    Note: Uses Motor (not PyMongo Async) for Qt/qasync compatibility.
    PyMongo's AsyncMongoClient doesn't work with external event loops.
    """
    
    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """
        Get DatabaseManager instance from ServiceLocator.
        
        Returns:
            DatabaseManager instance
            
        Raises:
            KeyError: If DatabaseManager not registered
        """
        from ..locator import sl
        return sl.get_system(cls)
    
    def __init__(self, locator: 'ServiceLocator', config):
        super().__init__(locator, config)
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def initialize(self):
        """Connect to database using config."""
        try:
            # Get config or use defaults
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
            
            # CRITICAL: Tell Motor to use the current event loop (qasync)
            # Without this, Motor creates its own loop causing "attached to different loop" errors
            import asyncio
            current_loop = asyncio.get_running_loop()
            self.client = AsyncIOMotorClient(conn_str, io_loop=current_loop)
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
        """
        Get a MongoDB collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            PyMongo async collection instance
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call initialize() first.")
        return self.db[name]
    async def emit_db_event(self, event_name: str, data: dict) -> None:
        """
        Helper to emit events from ORM layer which doesn't have direct access to EventBus.
        """
        from ..events import EventBus
        try:
            bus = self.locator.get_system(EventBus)
            await bus.publish(event_name, data)
        except (KeyError, Exception) as e:
            # Silent failure if event bus not available (during startup/shutdown tests)
            pass
