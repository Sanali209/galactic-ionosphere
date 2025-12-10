import os
from loguru import logger
from src.core.files.images import FileHandlerFactory
from src.core.database.models.task import TaskRecord

class ThumbnailService:
    """
    Handles thumbnail generation.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_thumb_path(self, content_hash: str) -> str:
        # Sharding: cache/ab/cd/hash.jpg
        shard1 = content_hash[:2]
        shard2 = content_hash[2:4]
        folder = os.path.join(self.cache_dir, shard1, shard2)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, f"{content_hash}.jpg")

    async def generate_task_handler(self, task: TaskRecord) -> dict:
        """
        Handler for 'GENERATE_THUMBNAIL' task.
        Payload: {"source_path": str, "content_hash": str}
        """
        src_path = task.payload.get("source_path")
        content_hash = task.payload.get("content_hash")
        
        if not src_path or not content_hash:
            raise ValueError("Missing path or hash")

        target_path = self.get_thumb_path(content_hash)
        
        if os.path.exists(target_path):
            return {"cached": True, "path": target_path}
            
        # Get Handler
        ext = os.path.splitext(src_path)[1]
        handler = FileHandlerFactory.get_handler(ext)
        if not handler:
            raise ValueError(f"No handler for {ext}")
            
        await handler.generate_thumbnail(src_path, target_path)
        return {"cached": False, "path": target_path}
