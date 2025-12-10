import asyncio
from typing import List, Dict
from loguru import logger
from src.core.database.models.detection import Detection
from src.core.database.models.image import ImageRecord

class ObjectDetectionService:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self._model = None
        self._loaded = False
        
    async def load(self):
        """Lazy load YOLO model."""
        if self._loaded:
            return
            
        logger.info(f"Loading Detection Model: {self.model_name}")
        # Run blocking load in thread
        await asyncio.to_thread(self._load_sync)
        self._loaded = True
        
    def _load_sync(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_name)
        
    async def detect(self, image_path: str) -> List[Dict]:
        """
        Run detection on an image.
        Returns list of dicts: {box: [x,y,w,h], label: str, conf: float}
        """
        if not self._loaded:
            await self.load()
            
        return await asyncio.to_thread(self._detect_sync, image_path)
        
    def _detect_sync(self, image_path: str) -> List[Dict]:
        results = self._model(image_path, verbose=False)
        output = []
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # box.xywhn returns [x_center, y_center, w, h] normalized
                # We typically want [x, y, w, h] normalized (top-left) for QML
                # YOLO returns xywh or xyxy. Let's get xywh normalized (center)
                # and convert to top-left if needed.
                # Actually, simpler: box.xyxyn (top-left, bottom-right normalized)
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                w = x2 - x1
                h = y2 - y1
                
                cls_id = int(box.cls[0])
                label = self._model.names[cls_id]
                conf = float(box.conf[0])
                
                output.append({
                    "box": [x1, y1, w, h],
                    "label": label,
                    "conf": conf
                })
        return output
