"""
UCoreFS - Detection Service

Manages object detection on files with configurable backends.
Integrates with ProcessingPipeline phase 3.
"""
from typing import List, Optional, Dict, Any, Type
from abc import ABC, abstractmethod
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.detection.models import DetectionInstance, DetectionClass


class DetectionBackend(ABC):
    """
    Abstract base class for detection backends.
    
    Implementations: YOLO, MTCNN, etc.
    """
    
    name: str = "base"
    
    @abstractmethod
    async def detect(self, image_path: str, settings: Dict[str, Any] = None) -> List[Dict]:
        """
        Run detection on image.
        
        Args:
            image_path: Path to image file
            settings: Backend-specific settings
            
        Returns:
            List of detection dicts with:
            - label: str (e.g., "face", "person")
            - bbox: dict with x, y, w, h (normalized 0-1)
            - confidence: float
            - image: Optional cropped PIL image
        """
        pass
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for this backend."""
        return {}


class DetectionService(BaseSystem):
    """
    Object detection service using YOLO/MTCNN.
    
    Provides face detection and object detection capabilities.
    Results are stored in Detection collection.
    """
    
    depends_on = []  # Independent service
    
    def __init__(self, locator, config):
        """
        Initialize DetectionService.
        
        Args:
            locator: ServiceLocator instance
            config: ConfigManager instance
        """
        super().__init__(locator, config)
        self._backends: Dict[str, DetectionBackend] = {}
        self._default_backend = "yolo"
    
    async def initialize(self):
        """Initialize detection service and load configured backends."""
        await super().initialize()
        logger.info("DetectionService initializing...")
        
        # Load backends based on config
        config = self.config
        try:
            detection_config = config.data.processing.detection
        except AttributeError:
            logger.warning("Detection config not found, using defaults")
            detection_config = None
        
        if detection_config:
            if detection_config.yolo.enabled:
                self._load_yolo_backend(detection_config.yolo.model_dump())
            
            if detection_config.mtcnn.enabled:
                self._load_mtcnn_backend(detection_config.mtcnn.model_dump())
        
        logger.info(f"DetectionService initialized with backends: {list(self._backends.keys())}")
    
    async def shutdown(self):
        """Shutdown detection service."""
        logger.info("DetectionService shutting down")
        self._backends.clear()
        await super().shutdown()
    
    def register_backend(self, backend: DetectionBackend):
        """
        Register a detection backend.
        
        Args:
            backend: DetectionBackend instance
        """
        self._backends[backend.name] = backend
        logger.info(f"Registered detection backend: {backend.name}")
        
    def get_backend(self, name: str) -> Optional[DetectionBackend]:
        """Get backend by name."""
        return self._backends.get(name)
    
    def _load_yolo_backend(self, settings: Dict[str, Any]):
        """Load YOLO backend if available."""
        try:
            from src.ucorefs.detection.backends.yolo import YOLOBackend
            backend = YOLOBackend(settings)
            self.register_backend(backend)
        except ImportError as e:
            logger.warning(f"YOLO backend not available: {e}")
    
    def _load_mtcnn_backend(self, settings: Dict[str, Any]):
        """Load MTCNN backend if available."""
        try:
            from src.ucorefs.detection.backends.mtcnn import MTCNNBackend
            backend = MTCNNBackend(settings)
            self.register_backend(backend)
        except ImportError as e:
            logger.warning(f"MTCNN backend not available: {e}")
    
    async def detect(
        self, 
        file_id: ObjectId, 
        backend: str = None,
        save: bool = True
    ) -> List[DetectionInstance]:
        """
        Run detection on a file.
        
        Args:
            file_id: FileRecord ObjectId
            backend: Backend name (default: configured default)
            save: Whether to save detections to database
            
        Returns:
            List of DetectionInstance objects
        """
        from src.ucorefs.models import FileRecord
        
        backend_name = backend or self._default_backend
        if backend_name not in self._backends:
            logger.warning(f"Backend '{backend_name}' not available")
            return []
        
        # Get file record
        file_record = await FileRecord.get(file_id)
        if not file_record:
            logger.warning(f"File not found: {file_id}")
            return []
        
        # Run detection
        detector = self._backends[backend_name]
        try:
            raw_detections = await detector.detect(file_record.path)
        except Exception as e:
            logger.error(f"Detection failed for {file_id}: {e}")
            return []
        
        # Convert to DetectionInstance records
        instances = []
        for det in raw_detections:
            instance = DetectionInstance(
                name=f"{det['label']}_{len(instances)}",
                file_id=file_id,
                bbox=det.get("bbox", {}),
                confidence=det.get("confidence", 0.0),
            )
            
            # Get or create DetectionClass for this label
            detection_class = await self._get_or_create_class(det["label"])
            if detection_class:
                instance.detection_class_id = detection_class._id
            
            if save:
                await instance.save()
            
            instances.append(instance)
        
        logger.info(f"Detected {len(instances)} objects in file {file_id}")
        return instances
    
    async def get_detections(self, file_id: ObjectId) -> List[DetectionInstance]:
        """
        Get existing detections for a file.
        
        Args:
            file_id: FileRecord ObjectId
            
        Returns:
            List of DetectionInstance objects
        """
        return await DetectionInstance.find({"file_id": file_id})
    
    async def _get_or_create_class(self, label: str) -> Optional[DetectionClass]:
        """Get or create DetectionClass for label."""
        existing = await DetectionClass.find_one({"class_name": label})
        if existing:
            return existing
        
        # Create new class
        detection_class = DetectionClass(
            name=label,
            class_name=label
        )
        await detection_class.save()
        return detection_class
        
    async def update_file_detections(self, file_id: ObjectId, detections: List[Dict[str, Any]]) -> bool:
        """
        Update/Replace all detections for a file.
        
        Args:
            file_id: File ID to update
            detections: List of detection dicts (label, bbox, confidence)
            
        Returns:
            True if successful
        """
        try:
            # 1. Delete existing detections
            # Note: This removes all history!
            await DetectionInstance.delete_many({"file_id": file_id})
            
            # 2. Create new ones
            for i, det in enumerate(detections):
                label = det.get("label", "Unknown")
                bbox = det.get("bbox", {})
                confidence = det.get("confidence", 1.0) # User added = 100% confidence?
                
                instance = DetectionInstance(
                    name=f"{label}_{i}",
                    file_id=file_id,
                    bbox=bbox,
                    confidence=confidence
                )
                
                # Resolve class
                detection_class = await self._get_or_create_class(label)
                if detection_class:
                    instance.detection_class_id = detection_class._id
                
                await instance.save()
            
            # 3. Update FileRecord.detections cache if it exists?
            # Current implementation of search uses Aggregation on DetectionInstance, 
            # so updating FileRecord might not be strictly necessary unless we double-write.
            # But let's check if FileRecord acts as cache.
            # FileRecord usually stores a summary or just has 'detections' field if using simplistic extraction.
            # But here we are using full DetectionInstance model.
            
            return True
        except Exception as e:
            logger.error(f"Failed to update detections for {file_id}: {e}")
            return False
