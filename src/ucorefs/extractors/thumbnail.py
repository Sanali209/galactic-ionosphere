"""
UCoreFS - Thumbnail Extractor

Generates and caches thumbnails for files.
Phase 2 extractor - runs in batches of 20.
"""
from typing import List, Dict, Any
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class ThumbnailExtractor(Extractor):
    """
    Generates thumbnails for image/video files.
    
    Wraps ThumbnailService for use in ProcessingPipeline.
    """
    
    name = "thumbnail"
    phase = 2
    priority = 100  # High priority - thumbnails first
    batch_supported = True
    is_cpu_heavy = True  # SAN-14: PIL operations (already thread-offloaded in ThumbnailService)
    needs_model = True   # Needs ThumbnailService from Engine locator
    
    # Supported file types for thumbnails
    SUPPORTED_TYPES = {"image", "video", "pdf"}
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Generate thumbnails for files.
        
        Args:
            files: List of files to process
            
        Returns:
            Dict mapping file_id -> thumbnail bytes
        """
        from src.ucorefs.thumbnails import ThumbnailService
        
        results = {}
        
        # Get thumbnail service
        if not self.locator:
            return results
            
        try:
            thumb_service = self.locator.get_system(ThumbnailService)
        except KeyError:
            logger.warning("ThumbnailService not available")
            return results
        
        for file in files:
            if not self.can_process(file):
                continue
                
            try:
                # Generate thumbnail (ThumbnailService handles caching)
                thumbnail = await thumb_service.get_or_create(file._id, size=256)
                if thumbnail:
                    results[file._id] = thumbnail
            except Exception as e:
                logger.error(f"Thumbnail generation failed for {file._id}: {e}")
        
        return results
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """
        Mark file as having thumbnail.
        
        ThumbnailService already caches the bytes; we just update the flag.
        """
        try:
            file = await FileRecord.get(file_id)
            if file:
                file.has_thumbnail = True
                if file.processing_state < ProcessingState.THUMBNAIL_READY:
                    file.processing_state = ProcessingState.THUMBNAIL_READY
                await file.save()
                return True
        except Exception as e:
            logger.error(f"Failed to update thumbnail flag for {file_id}: {e}")
        return False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image/video/pdf files."""
        return file.file_type in self.SUPPORTED_TYPES
