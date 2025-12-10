import os
import hashlib
import asyncio
from typing import Optional
from loguru import logger

from src.core.locator import sl
from src.core.files.images import FileHandlerFactory
from src.core.database.models.image import ImageRecord

class ImportService:
    """
    Coordinates file ingestion: Hash -> DB -> Background Tasks.
    """
    
    async def process_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        handler = FileHandlerFactory.get_handler(ext)
        
        if not handler:
            # Unsupported file
            return

        logger.info(f"Processing file: {file_path}")
        
        # 1. Hashing (Async wrapper if strictly needed, but fast IO is usually OK-ish for small batches. 
        # For huge files, we'd offload to thread/task.)
        # Here we do it inline for simplicity or offload?
        # Let's offload to asyncio.to_thread for responsiveness.
        content_hash = await asyncio.to_thread(self._calculate_hash, file_path)
        
        # 2. Key Check: Does hash exist?
        existing = await ImageRecord.find({"content_md5": content_hash})
        if existing:
            logger.info(f"Duplicate image found: {file_path} (Matches {existing[0].id})")
            # Logic: Add a 'Reference' (DUPLICATE_OF)? Or just skip?
            # For Phase 2, we just ensure we have *a* record.
            # If path is different, maybe update or track multiple paths?
            # Current ImageRecord has one 'path'. 
            # We'll skip re-importing identical existing entity.
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
        
        await record.save()
        logger.info(f"Imported image: {record.id}")
        
        # 4. Dispatch Tasks
        # We need a reference to the dispatcher. 
        # Using ServiceLocator pattern as strictly requested in architecture.
        # But 'sl' might not have 'tasks' if not initialized.
        # Assuming we can grab it or we inject it properly.
        # Let's import the global 'sl' or assume 'engine.tasks' is available.
        # Ideally ImportService takes dispatcher in __init__.
        # For now, let's use the one from `src.core.engine.tasks` if we made it singleton?
        # No, 'TaskDispatcher' is a class.
        # Let's access it via `sl.tasks` (assuming we add it to SL).
        
        # For now, let's just create a quick connection or assume `self.dispatcher` is set.
        # Better: ImportService(dispatcher).
        
        if self.dispatcher:
            await self.dispatcher.submit_task("GENERATE_THUMBNAIL", {
                "source_path": file_path,
                "content_hash": content_hash
            })
        
        return record

    def __init__(self, dispatcher=None):
        self.dispatcher = dispatcher

    def _calculate_hash(self, path: str) -> str:
        """MD5 Hash of file content."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
