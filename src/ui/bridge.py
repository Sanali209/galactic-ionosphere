from PySide6.QtCore import QObject, Slot, Signal, Property
from src.core.engine.importer import ImportService
from src.core.ai.search import SearchService
from src.ui.models.grid_model import GalleryGridModel
import asyncio
from loguru import logger
from src.core.locator import sl
from src.core.database.models.image import ImageRecord
from src.core.database.models.tag import TagManager
from src.core.files.images import FileHandlerFactory

class BackendBridge(QObject):
    """
    Bridge between QML and Python Services.
    """
    
    # Signals
    searchFinished = Signal(int) # Count
    logMessage = Signal(str)
    # id, path, dimensions, size, meta
    imageSelected = Signal(str, str, str, str, str)
    # id, key, value, success
    metaUpdateResult = Signal(str, str, str, bool)
    # Property for AI result limit
    aiResultLimitChanged = Signal(int)
    _aiResultLimit = 20
    # Removed duplicate _ai_result_limit variable
    
    def __init__(self, importer, search_service, grid_model, journal_model, file_system_model, tag_model):
        super().__init__()
        self._importer = importer
        self._search_service = search_service
        self._grid_model = grid_model
        self._journal_model = journal_model
        self._model_file_system = file_system_model
        self._tag_model = tag_model
        # AI limit is now pulled from config on demand
        
        # Hook into Loguru? 
        # For now, we can manually emit log messages or use a sink.
        logger.add(lambda msg: self.logMessage.emit(str(msg).strip()), format="{time:HH:mm:ss} | {level} | {message}")

    @Property(QObject, constant=True)
    def journalModel(self):
        return self._journal_model

    @Slot()
    def refreshGallery(self):
        """Manually trigger gallery refresh from DB."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._refresh_gallery())

    async def _refresh_gallery(self):
        logger.info("Refreshing Gallery from DB...")
        images = await ImageRecord.find({})
        self._grid_model.set_images(images)
        # Also refresh file system tree
        await self._model_file_system.load_roots()
        await self._tag_model.load_tags()
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
        await self._refresh_gallery()
        
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

    @Slot(str)
    def filterByTag(self, tag_id: str):
        """Filter images by exact tag ID."""
        logger.info(f"Filter by Tag ID: {tag_id}")
        loop = asyncio.get_event_loop()
        
        async def _do_filter():
            from bson import ObjectId
            from src.core.database.models.image import ImageRecord
            try:
                # Query DB for images with this tag
                # Assuming tag_ids stored as ObjectIds list
                oid = ObjectId(tag_id)
                records = await ImageRecord.find({"tag_ids": oid}).to_list()
                logger.info(f"Found {len(records)} images for tag {tag_id}")
                
                # Update Gallery Model
                self._gallery_model.update_images(records)
            except Exception as e:
                logger.error(f"Filter failed: {e}")
                
        loop.create_task(_do_filter())

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

    @Slot(str, str, str)
    def updateImageMetadata(self, image_id: str, key: str, value: str):
        """
        Updates metadata for a specific image.
        key: 'rating', 'label', 'description', 'tags' (comma sep)
        value: string representation
        """
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_update_metadata(image_id, key, value))

    async def _do_update_metadata(self, image_id: str, key: str, value: str):
        logger.info(f"Updating metadata {image_id}: {key}={value}")
        record = await ImageRecord.get(image_id)
        if not record:
            logger.error("Image not found")
            self.metaUpdateResult.emit(image_id, key, value, False)
            return

        # Prepare update payload
        # Convert value based on key
        payload = {}
        if key == "rating":
            try:
                payload["rating"] = int(value)
            except:
                logger.error("Invalid rating value")
                return
        elif key == "tags":
            # Assume comma separated paths or names
            # Logic: If user provides "A|B, C", we parse two tags.
            payload["tags"] = [t.strip() for t in value.split(",") if t.strip()]
        else:
            payload[key] = value

        # 1. Update File
        ext = record.ext
        handler = FileHandlerFactory.get_handler(ext)
        if handler:
            try:
                await handler.write_metadata(record.full_path, payload)
            except Exception as e:
                logger.error(f"Failed to write metadata to file: {e}")
                self.metaUpdateResult.emit(image_id, key, value, False)
                return
        else:
             logger.warning(f"No handler for {ext}")

        # 2. Update DB
        # Re-read file or just update DB field?
        # Updating DB field mirrors the change.
        # But ideally we re-extract to be sure.
        # For responsiveness, update DB blindly, then maybe queue a re-read.

        # Mapping to DB fields
        if key == "rating":
            record.rating = int(value)
        elif key == "label":
            record.label = value
        elif key == "description":
            record.description = value
        # For tags, we sync with Tag entities immediately to keep DB graph correct
        if key == "tags" and "tags" in payload:
            tag_ids = []
            for tag_path in payload["tags"]:
                try:
                    # Treat input as hierarchical paths (User UI should likely provide paths)
                    leaf = await TagManager.ensure_from_path(tag_path, separator="|")
                    tag_ids.append(leaf.id)
                except Exception as e:
                    logger.error(f"Failed to reconcile tag {tag_path}: {e}")
            record.tag_ids = tag_ids

        # Update xmp_data cache in record
        if not record.xmp_data: record.xmp_data = {}
        # Merge changes
        # This is rough, as xmp_data structure varies.
        # Ideally we re-run extract_metadata

        try:
            new_meta = await handler.extract_metadata(record.full_path)
            record.xmp_data = new_meta
            # Sync root fields
            if "rating" in new_meta: record.rating = new_meta["rating"]
            if "label" in new_meta: record.label = new_meta["label"]
            if "description" in new_meta: record.description = new_meta["description"]

            await record.save()
            self.metaUpdateResult.emit(image_id, key, value, True)

            # Notify UI of change via selection update if selected?
            # Or just let the signal handle it.

        except Exception as e:
            logger.error(f"Failed to refresh DB record: {e}")
            self.metaUpdateResult.emit(image_id, key, value, False)

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
    @Slot()
    def wipeDb(self):
        """Wipes the database and clears the gallery."""
        logger.warning("Wiping Database requested by User.")
        loop = asyncio.get_event_loop()
        loop.create_task(self._do_wipe_db())

    async def _do_wipe_db(self):
        print("DEBUG: _do_wipe_db executing")
        from src.core.database.manager import db_manager
        await db_manager.reset_db()
        self._grid_model.set_images([])
        self.logMessage.emit("Database wiped. Please restart or re-import.")
        logger.info("Database wiped.")
