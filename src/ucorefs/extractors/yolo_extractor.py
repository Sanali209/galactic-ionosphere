"""
UCoreFS - YOLO Extraction
"""
from typing import List, Dict, Any
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.models.file_record import FileRecord

class YOLOExtractor(AIExtractor):
    """
    YOLO Object Detection Extractor (Phase 3).
    
    Uses DetectionService -> YOLOBackend to find objects and save them
    as DetectionInstance records.
    """
    
    name = "yolo"
    phase = 3
    priority = 80 # High priority in Phase 3
    batch_supported = False
    needs_model = True  # Requires YOLO model via DetectionService
    
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
                return True
            except Exception as e:
                logger.error(f"[YOLOExtractor] Failed to get DetectionService: {e}")
                return False
        return False

    def can_process(self, file: FileRecord) -> bool:
        return file.file_type in self.SUPPORTED_TYPES
    
    # Inherited from AIExtractor: _get_llm_service(), extract()
    
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """Extract detections using LLMWorkerService."""

        from src.core.llm.models import LLMJobRequest
        
        file_paths = [str(f.path) for f in files if self.can_process(f)]
        path_to_id = {str(f.path): f.id for f in files}
        
        request = LLMJobRequest(
            task_type="yolo",
            file_paths=file_paths,
            options={"classes": self._class_filter} if self._class_filter else {}
        )
        
        try:
            future = await llm_service.submit_job(request)
            result = await future
            
            if not result.success:
                logger.warning(f"YOLO worker job failed: {result.error}, falling back to legacy")
                return await self._extract_legacy(files)
            
            # Check if worker returned actual data
            results = {}
            has_real_data = False
            for file_path, data in (result.data or {}).items():
                file_id = path_to_id.get(file_path)
                if file_id:
                    results[file_id] = data
                    if data is not None:
                        has_real_data = True
            
            # If worker returned all None (stub), fall back to legacy
            if not has_real_data:
                logger.debug("[YOLO] Worker returned empty data, falling back to legacy")
                return await self._extract_legacy(files)
            
            logger.info(f"[YOLO] Processed {len(results)}/{len(files)} via LLMWorkerService")
            return results
            
        except Exception as e:
            logger.error(f"YOLO extraction via service failed: {e}")
            return await self._extract_legacy(files)
    
    async def _extract_legacy(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Legacy: Extract using DetectionService."""
        results = {}
        
        if not await self._ensure_service():
            return results
            
        for file in files:
            try:
                logger.info(f"[YOLOExtractor] Processing file: {file.path}")
                
                detection_instances = await self._service.detect(
                    file_id=file.id, 
                    backend="yolo",
                    save=False
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
            # Empty list (0 detections) is a valid result, not an error
            if not result:
                logger.debug(f"[YOLOExtractor] No detections to store for {file_id}")
                return True  # Success - just nothing to save
                
            logger.info(f"[YOLOExtractor] Storing {len(result)} detections for {file_id}")
            
            for instance in result:
                await instance.save()
                
            return True
        except Exception as e:
            logger.error(f"[YOLOExtractor] Failed to store results: {e}")
            return False
