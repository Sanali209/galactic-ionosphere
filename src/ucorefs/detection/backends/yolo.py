"""
UCoreFS Detection - YOLO Backend

YOLOv8/YOLO11 object detection backend using ultralytics.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from src.ucorefs.detection.service import DetectionBackend


class YOLOBackend(DetectionBackend):
    """
    YOLO object detection backend.
    
    Uses ultralytics YOLO models (v8/v11) for detection.
    
    Settings:
        model: Model name or path (default: "yolov8n.pt")
        confidence: Min confidence threshold (default: 0.25)
        classes: Optional list of class names to filter
        use_gpu: Whether to use GPU (default: True)
    """
    
    name = "yolo"
    
    def __init__(self, settings: Dict[str, Any] = None):
        """
        Initialize YOLO backend.
        
        Args:
            settings: Backend configuration
        """
        self._settings = settings or {}
        self._model = None
        self._model_path = self._settings.get("model", "yolov8n.pt")
        self._confidence = self._settings.get("confidence", 0.25)
        self._classes = self._settings.get("classes", None)
        self._use_gpu = self._settings.get("use_gpu", True)
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            
            self._model = YOLO(self._model_path)
            device = "cuda" if self._use_gpu else "cpu"
            
            # Warm up model
            logger.info(f"YOLO backend loaded: {self._model_path} on {device}")
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            import traceback
            logger.error(f"Failed to load YOLO model: {e}\n{traceback.format_exc()}")
            raise
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default YOLO settings."""
        return {
            "model": "yolov8n.pt",
            "confidence": 0.25,
            "classes": None,
            "use_gpu": True
        }
    
    async def detect(self, image_path: str, settings: Dict[str, Any] = None) -> List[Dict]:
        """
        Run YOLO detection on image in a thread-safe manner.
        
        Args:
            image_path: Path to image
            settings: Optional override settings
            
        Returns:
            List of detection dicts
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # Offload blocking inference to thread pool
        # We use a lambda or partial to pass arguments to the synchronous method
        return await loop.run_in_executor(None, self._detect_sync, image_path, settings)

    def _detect_sync(self, image_path: str, settings: Dict[str, Any] = None) -> List[Dict]:
        """
        Synchronous detection method to run in worker thread.
        """
        if self._model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        # Merge settings
        conf = settings.get("confidence", self._confidence) if settings else self._confidence
        classes = settings.get("classes", self._classes) if settings else self._classes
        
        # Check CUDA availability at runtime, not just config
        import torch
        device = "cuda" if (self._use_gpu and torch.cuda.is_available()) else "cpu"
        
        try:
            # Run inference
            results = self._model(
                image_path,
                conf=conf,
                device=device,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                # Get image dimensions for normalization
                img_h, img_w = result.orig_shape
                
                for i, box in enumerate(boxes):
                    # Get class name
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    
                    # Filter by class if specified
                    if classes and label not in classes:
                        continue
                    
                    # Get bounding box (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Normalize to 0-1 range
                    bbox = {
                        "x": x1 / img_w,
                        "y": y1 / img_h,
                        "w": (x2 - x1) / img_w,
                        "h": (y2 - y1) / img_h
                    }
                    
                    detections.append({
                        "label": label,
                        "bbox": bbox,
                        "confidence": float(box.conf[0]),
                    })
            
            logger.debug(f"YOLO detected {len(detections)} objects in {image_path}")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
