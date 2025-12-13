import os
import hashlib
import asyncio
from typing import Optional
from loguru import logger

from src.core.locator import sl
from src.core.files.images import FileHandlerFactory
from src.core.database.models.image import ImageRecord
from src.core.database.models.tag import TagManager
from src.core.database.models.folder import FolderRecord

class ImportService:
    """
    Coordinates file ingestion: Hash -> DB -> Background Tasks.
    """
    
    def __init__(self, dispatcher, embed_service, vector_driver):
        self._dispatcher = dispatcher
        self.embed_service = embed_service
        self.vector_driver = vector_driver
        self._folder_cache = set()

    async def _ensure_folder(self, path: str):
        # Normalize: Lowercase for Windows consistency, Forward slashes
        path = path.replace("\\", "/").rstrip("/").lower()
        if not path: return
        
        if path.endswith(":"): path += "/" # Restore drive root if needed (e.g. C:)
        
        if path in self._folder_cache: return
        self._folder_cache.add(path) # Optimistic locking for this run

        # Check DB
        exists = await FolderRecord.find_one({"path": path})
        if not exists:
            # Create
            import os
            name = os.path.basename(path)
            # Detect drive root carefully
            # os.path.dirname returns native separators. 
            parent = os.path.dirname(path).replace("\\", "/").rstrip("/").lower()
            if parent.endswith(":"): parent += "/"
            
            # If path was "d:/", parent is "d:/".
            
            if parent == path:
                parent = None
            else:
                 await self._ensure_folder(parent)

            try:
                # Re-check to be safe from other processes or async race
                exists_check = await FolderRecord.find_one({"path": path})
                if not exists_check:
                    rec = FolderRecord(path=path, name=name or path, parent_path=parent)
                    await rec.save()
                    logger.info(f"Created FolderRecord: {path}")
            except Exception as e:
                logger.error(f"Error creating folder {path}: {e}")

    async def process_file(self, file_path: str):
        # 0. Sync Folder Record
        import os
        parent_dir = os.path.dirname(file_path)
        await self._ensure_folder(parent_dir)

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        handler = FileHandlerFactory.get_handler(ext)
            
        if not handler:
            # Unsupported file
            return
 
        logger.info(f"Processing file: {file_path}")
        
        # 1. Hashing
        content_hash = await asyncio.to_thread(self._calculate_hash, file_path)
        
        # 2. Key Check: Does hash exist?
        existing = await ImageRecord.find({"content_md5": content_hash})
        if existing:
            logger.info(f"Duplicate image found: {file_path} (Matches {existing[0].id})")
            return existing[0]

        # 3. Create Record
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Extract Metadata
        meta = await handler.extract_metadata(file_path)
        dims = await handler.get_dimensions(file_path)
        
        record = ImageRecord(path=file_dir, filename=file_name)
        record.ext = ext
        record.content_md5 = content_hash
        record.size_bytes = file_size
        record.xmp_data = meta
        record.dimensions = dims
        
        # 3b. Process Tags (Hierarchical)
        if "tags" in meta and meta["tags"]:
            tag_ids = []
            for tag_path in meta["tags"]:
                try:
                    # ensure_from_path handles creation and finding
                    leaf_tag = await TagManager.ensure_from_path(tag_path, separator="|")
                    tag_ids.append(leaf_tag.id)
                except Exception as e:
                    logger.error(f"Failed to process tag path '{tag_path}': {e}")
            record.tag_ids = tag_ids

        await record.save()
        logger.info(f"Imported image: {record.id}")
        
        # 4. Dispatch Tasks (Thumbnail)
        if self._dispatcher:
            await self._dispatcher.submit_task("GENERATE_THUMBNAIL", {
                "source_path": file_path,
                "content_hash": content_hash
            })

        # 5. Vectorize (Inline/Background)
        if self.embed_service and self.vector_driver:
            try:
                vec = await asyncio.to_thread(self.embed_service.generate_embedding, file_path)
                if vec is not None and len(vec) > 0:
                    payload = {"mongo_id": str(record.id), "path": record.full_path}
                    q_id = self.vector_driver.to_qdrant_id(str(record.id))
                    await self.vector_driver.upsert_vector(
                        point_id=q_id, 
                        vector=vec.tolist(),
                        payload=payload
                    )
                    logger.info(f"Vectorized {record.id}")
            except Exception as e:
                logger.error(f"Vectorization failed for {file_path}: {e}")
        
        return record

    def _calculate_hash(self, path: str) -> str:
        """MD5 Hash of file content."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
