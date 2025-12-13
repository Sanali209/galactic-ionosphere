from loguru import logger
from datetime import datetime
import asyncio
from src.core.journal.models import JournalRecord

class JournalService:
    def __init__(self):
        self._sink_id = None

    async def _async_save_journal(self, j_rec):
        """Helper to save record in async context"""
        try:
            await j_rec.save()
        except RuntimeError:
            pass # Loop closed?

    def set_ui_callback(self, callback):
        self._ui_callback = callback

    def _sink_handler(self, message):
        """
        Synchronous handler running in Loguru's worker thread.
        Dispatches async save to the main event loop safely.
        """
        try:
            record = message.record
            if "journal" in record["extra"]: return

            # 1. Prepare Record Object (Memory)
            j_rec = JournalRecord(
                timestamp=record["time"],
                level=record["level"].name,
                category="SYSTEM",
                message=record["message"],
                details={
                    "file": record["file"].name,
                    "line": record["line"],
                    "exception": str(record["exception"]) if record["exception"] else None
                }
            )

            # 2. Send to UI (Always, if callback exists)
            # Signal/Callback should be thread-safe (Qt handles signals from threads ok, callbacks maybe not)
            if self._ui_callback:
                self._ui_callback(j_rec)

            # 3. Store in DB (Only if Warning/Error)
            if record["level"].no >= logger.level("WARNING").no:
                 # We need the main loop to run the async save
                 if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self._async_save_journal(j_rec), self._loop)

        except Exception as e:
            # Avoid logging here to prevent recursion
            print(f"FAILED TO WRITE TO JOURNAL: {e}")

    def start(self):
        """
        Registers the loguru sink to capture WARNING/ERROR logs automatically.
        """
        # Capture the loop where start() is called (Main Loop)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
            logger.warning("JournalService started without active event loop, DB persistance might fail.")

        # Filter: Level >= WARNING
        self._sink_id = logger.add(
            self._sink_handler,
            level="INFO", 
            serialize=False,
            enqueue=True 
        )
        logger.info("Journal Service Started (MongoDB Sink + UI Attached)")
        
        self._ui_callback = None

    async def log_event(self, category: str, message: str, level: str = "INFO", details: dict = None):
        """
        Explicitly log a structured event.
        """
        try:
            j_rec = JournalRecord(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                details=details or {}
            )
            await j_rec.save()
            
            # Also echo to console/file if it's important
            # We add extra={"journal": True} to prevent double logging in sink if level is high
            logger.bind(journal=True).log(level, f"[{category}] {message}")
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
