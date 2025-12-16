from datetime import datetime
from loguru import logger
from ..base_system import BaseSystem
from .models import JournalEntry

class JournalService(BaseSystem):
    """
    System for recording application events to the database.
    """
    async def initialize(self):
        logger.info("JournalService initialized.")
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()

    async def log(self, level: str, source: str, message: str, details: dict = None):
        """
        Logs an event to the database and Loguru.
        """
        # 1. Log to console/file via Loguru
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{source}] {message}")

        # 2. Log to DB
        entry = JournalEntry(
            timestamp=datetime.utcnow(),
            level=level,
            source=source,
            message=message,
            details=details
        )
        try:
            await entry.save()
        except Exception as e:
            logger.error(f"Failed to save journal entry: {e}")
            
    async def get_recent(self, limit=50):
        # Todo: Implement sorting and limiting in ORM find
        # For now, just return all (inefficient, but POC)
        return await JournalEntry.find({})
