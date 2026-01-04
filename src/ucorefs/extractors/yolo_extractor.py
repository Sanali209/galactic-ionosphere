"""
UCoreFS - YOLO Extraction
"""
from typing import List, Dict, Any
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord

class YOLOExtractor(Extractor):
    """
    YOLO Object Detection Extractor (Phase 3).
    
    Uses DetectionService -> YOLOBackend to find objects and save them
    as DetectionInstance records.
    """
    
    name = "yolo"
    phase = 3
    priority = 80 # High priority in Phase 3
    batch_supported = False
    
    SUPPORTED_TYPES = {"image"}
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._service = None
        self._class_filter = config.get("classes") if config else None
        
        if self._class_filter:
            logger.info(f"[YOLOExtractor] Initialized with class filter: {self._class_filter}")
        else:
            logger.info("[YOLOExtractor] Initialized with all YOLO classes (80 total)")
        
    async def _ensure_service(self):
        if self._service:
            return True
        if self.locator:
            try:
                from src.ucorefs.detection.service import DetectionService
                self._service = self.locator.get_system(DetectionService)
                # Ensure YOLO backend is active (this might need explicit configuration or init)
                # The service usually auto-loads configured backends.
                return True
            except Exception as e:
                logger.error(f"[YOLOExtractor] Failed to get DetectionService: {e}")
                return False
        return False

    def can_process(self, file: FileRecord) -> bool:
        return file.file_type in self.SUPPORTED_TYPES

    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        results = {}
        if not await self._ensure_service():
            return results
            
        for file in files:
            try:
                logger.info(f"[YOLOExtractor] Processing file: {file.path}")
                
                # DetectionService.detect() returns DetectionInstance objects
                # It takes file_id and backend name
                detection_instances = await self._service.detect(
                    file_id=file.id, 
                    backend="yolo",
                    save=False  # We'll save in store() to maintain extractor flow
                )
                
                logger.info(f"[YOLOExtractor] Found {len(detection_instances)} objects in {file.name}")
                results[file.id] = detection_instances
                
            except Exception as e:
                logger.error(f"[YOLOExtractor] Failed to process {file.path}: {e}")
                
        return results

    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store detection instances."""
        try:
            # Result is already a list of DetectionInstance objects from detect()
            # Just need to save them
            if not result:
                return False
                
            logger.info(f"[YOLOExtractor] Storing {len(result)} detections for {file_id}")
            
            for instance in result:
                await instance.save()
                
            return True
        except Exception as e:
            logger.error(f"[YOLOExtractor] Failed to store results: {e}")
            return False
