import asyncio
import time
from typing import Callable
from loguru import logger
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    logger.warning("Watchdog not installed. File monitoring disabled.")
    Observer = None
    FileSystemEventHandler = object

class galleryEventHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None], loop):
        self.callback = callback
        self.loop = loop

    def on_created(self, event):
        if not event.is_directory:
            # Dispatch to async loop
            asyncio.run_coroutine_threadsafe(self.callback(event.src_path), self.loop)

class FileMonitor:
    """
    Watches directories for new files.
    """
    def __init__(self, callback: Callable[[str], None]):
        self.observer = Observer() if Observer else None
        self.callback = callback
        self.watches = {}

    def start(self):
        if not self.observer:
            return
        self.observer.start()
        logger.info("File Monitor started.")

    def stop(self):
        if not self.observer:
            return
        self.observer.stop()
        self.observer.join()

    def add_watch(self, path: str):
        if not self.observer:
            return
        if path in self.watches:
            return
            
        loop = asyncio.get_running_loop()
        handler = galleryEventHandler(self.callback, loop)
        watch = self.observer.schedule(handler, path, recursive=True)
        self.watches[path] = watch
        logger.info(f"Watching: {path}")
