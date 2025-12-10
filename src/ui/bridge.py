from PySide6.QtCore import QObject, Slot, Signal, Property
from src.core.engine.importer import ImportService
from src.core.ai.search import SearchService
from src.ui.models.grid_model import GalleryGridModel
import asyncio
from loguru import logger
from src.core.database.models.image import ImageRecord

class BackendBridge(QObject):
    """
        if path.startswith("/") and ":" in path: path = path[1:]
        
        # Normpath essential for comparison
        import os
        path = os.path.normpath(path)
        
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_filter_folder(path, recursive))
        
    async def _do_filter_folder(self, folder_path: str, recursive: bool):
        logger.info(f"Filtering by folder: {folder_path} (Recursive: {recursive})")
        
        # Regex for path matching
        # MongoDB regex: we want paths that start with folder_path
        # But image.path stores the *folder* path, image.filename stores name.
        # So finding images IN a folder is finding records where `path` == `folder_path`
        
        # Normalize DB paths if needed (stored with os.path.sep?)
        # Let's assume consistent separators for now or do regex.
        
        if recursive:
            # Starts with folder_path
            # Escape regex special chars in path
            import re
            escaped = re.escape(folder_path)
            query = {"path": {"$regex": f"^{escaped}"}}
        else:
            # Exact match
            query = {"path": folder_path}
            
        results = await ImageRecord.find(query)
        self._grid_model.set_images(results)
        self.searchFinished.emit(len(results))
        logger.info(f"Filter returned {len(results)} images.")

    async def _do_search(self, query: str):
        results = await self._search.search_by_text(query)
        self._grid_model.set_images(results)
        self.searchFinished.emit(len(results))
        logger.info(f"Found {len(results)} results.")
