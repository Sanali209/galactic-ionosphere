from PySide6.QtCore import QObject, Slot, Signal, Property
from src.core.engine.importer import ImportService
from src.core.ai.search import SearchService
from src.ui.models.grid_model import GalleryGridModel
import asyncio
from loguru import logger
from src.core.locator import sl
from src.core.database.models.image import ImageRecord

class BackendBridge(QObject):
    """
    Bridge between QML and Python Services.
    """
    
    # Signals
    searchFinished = Signal(int) # Count
    logMessage = Signal(str)
    # id, path, dimensions, size, meta
    imageSelected = Signal(str, str, str, str, str)
    # Property for AI result limit
    aiResultLimitChanged = Signal(int)
    _aiResultLimit = 20
    # Removed duplicate _ai_result_limit variable
    
    def __init__(self, importer: ImportService, search_service: SearchService, grid_model: GalleryGridModel):
        super().__init__()
        self._importer = importer
        self._search = search_service
        self._grid_model = grid_model
        # AI limit is now pulled from config on demand
        
        # Hook into Loguru? 
        # For now, we can manually emit log messages or use a sink.
        logger.add(lambda msg: self.logMessage.emit(str(msg).strip()), format="{time:HH:mm:ss} | {level} | {message}")

    @Slot()
    def refreshGallery(self):
        """Manually trigger gallery refresh from DB."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._refresh_gallery())

    async def _refresh_gallery(self):
        logger.info("Refreshing Gallery from DB...")
        images = await ImageRecord.find({})
        self._grid_model.set_images(images)
        self.logMessage.emit(f"Gallery refreshed: {len(images)} images.")
        logger.info(f"Gallery refreshed with {len(images)} images.")

    @Slot(str)
    def importFolder(self, path: str):
        """Trigger import of a folder (non-blocking in UI, async background)."""
        # Clean path (QML sends file:///...)
        clean_path = path.replace("file:///", "")
        # Windows patch: file:///C:/... becomes C:/... usually fine,
        # but if it was just /C:/ on some platforms, verify.
        # On Windows QML usually sends file:///D:/... -> D:/...
        # Simple fix:
        if clean_path.startswith("/") and ":" in clean_path:
             # e.g. /C:/Users...
            clean_path = clean_path[1:]
            
        logger.info(f"Importing folder: {clean_path}")
        
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_import_folder(clean_path))

    async def _do_import_folder(self, root_path: str):
        import os
        count = 0
        for root, dirs, files in os.walk(root_path):
            for file in files:
                full_path = os.path.join(root, file)
                # Filter useful extensions? Importer does check, but we can fast fail.
                await self._importer.process_file(full_path)
                count += 1
                if count % 10 == 0:
                     self.logMessage.emit(f"Imported {count} files...")
        
        self.logMessage.emit(f"Import Finished. Total: {count}")
        logger.info(f"Import Finished. Total: {count}")
        
        # Refresh Gallery
        await self._refresh_gallery()

    @Slot(str)
    def search(self, query: str):
        """Trigger semantic search."""
        logger.info(f"Search: {query}")
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_search(query))

    @Slot(str)
    def selectImage(self, image_id: str):
        """Called when user clicks an image in Grid/List."""
        logger.info(f"Selected: {image_id}")
        loop = asyncio.get_event_loop()
        loop.create_task(self._fetch_details(image_id))

    # Generic settings access
    @Slot(str, result=str)
    def getSetting(self, key: str) -> str:
        """Retrieve a setting value using dot notation, e.g. 'ai.provider_id'."""
        try:
            section, subkey = key.split('.')
            cfg = sl.config.data
            value = getattr(getattr(cfg, section), subkey)
            return str(value)
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return ""

    @Slot(str, str)
    def setSetting(self, key: str, value: str):
        """Update a setting using dot notation. Value is passed as string and converted if possible."""
        try:
            section, subkey = key.split('.')
            # Simple type inference: try int, float, else keep string
            if value.isdigit():
                cast_val = int(value)
            else:
                try:
                    cast_val = float(value)
                except ValueError:
                    cast_val = value
            sl.config.update(section, subkey, cast_val)
        except Exception as e:
            logger.error(f"Failed to set setting {key} to {value}: {e}")
        
    async def _fetch_details(self, image_id: str):
        record = await ImageRecord.get(image_id)
        if record:
            path = record.full_path
            dims = f"{record.dimensions.get('width', 0)}x{record.dimensions.get('height', 0)}"
            size_mb = f"{record.size_bytes / (1024*1024):.2f} MB"
            meta = str(record.xmp_data) # Send as string for now
            self.imageSelected.emit(str(record.id), path, dims, size_mb, meta)
        else:
            self.imageSelected.emit(image_id, "Unknown", "-", "-", "{}")

    @Slot(str, bool)
    def filterByFolder(self, folder_url: str, recursive: bool):
        """Filters gallery grid by folder path."""
        # Clean path
        path = folder_url.replace("file:///", "")
        if path.startswith("/") and ":" in path: path = path[1:]
        
        # Normpath essential for comparison
        import os
        path = os.path.normpath(path)
        
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_filter_folder(path, recursive))
        
    async def _do_filter_folder(self, folder_path: str, recursive: bool):
        logger.info(f"Filtering by folder: {folder_path} (Recursive: {recursive})")
        
        if recursive:
            # Starts with folder_path
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

    @Slot()
    def vectorizeAll(self):
        """Backfill vectors for all images."""
        logger.info("Starting Batch Vectorization...")
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_vectorize_all())

    async def _do_vectorize_all(self):
        # Very crude batch process
        all_images = await ImageRecord.find({})
        logger.info(f"Found {len(all_images)} images to check.")
        
        count = 0
        for img in all_images:
            # Check if exists in Qdrant? Or just overwrite?
            # For speed, let's just do it. 
            # Ideally we check a 'vectorized' flag in Mongo.
            
            # Use importer's logic or internal helper?
            # Importer has the logic but requires path.
            # We must access embedding service directly here or expose Importer method.
            # But Importer takes `process_file`.
            # Let's use `self._importer.embed_service` if we updated the instance passed.
            # Wait, `BackendBridge` gets `importer` which now has `embed_service`.
            
            svc = self._importer.embed_service
            drv = self._importer.vector_driver
            
            if not svc or not drv:
                logger.error("AI Services not available in Importer.")
                return

            try:
                vec = await asyncio.to_thread(svc.generate_embedding, img.full_path)
                if vec is not None and len(vec) > 0:
                     payload = {"mongo_id": str(img.id), "path": img.full_path}
                     q_id = drv.to_qdrant_id(str(img.id))
                     await drv.upsert_vector(q_id, vec.tolist(), payload)
                     count += 1
                     if count % 10 == 0:
                         self.logMessage.emit(f"Vectorized {count}/{len(all_images)}")
                         await asyncio.sleep(0.01) # Yield to UI
            except Exception as e:
                logger.error(f"Failed to vectorize {img.id}: {e}")
                
        self.logMessage.emit(f"Vectorization Complete. Processed {count} images.")
        logger.info(f"Vectorization Complete. Processed {count} images.")

    async def _do_search(self, query: str):
        limit = sl.config.data.ai.result_limit
        results = await self._search.search_by_text(query, limit=limit)
        self._grid_model.set_images(results)
        self.searchFinished.emit(len(results))
        logger.info(f"Found {len(results)} results.")

    @Property(int, notify=aiResultLimitChanged)
    def aiResultLimit(self):
        return sl.config.data.ai.result_limit

    @Slot(int)
    def setAiResultLimit(self, limit):
        """Set the maximum number of AI search results to return."""
        if limit > 0:
            sl.config.update("ai", "result_limit", limit)
            self.aiResultLimitChanged.emit(limit)
