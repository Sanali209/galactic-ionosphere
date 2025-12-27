"""
UCoreFS Detection - MTCNN Backend

MTCNN face detection backend using facenet-pytorch.
"""
from typing import List, Dict, Any
from loguru import logger

from src.ucorefs.detection.service import DetectionBackend


class MTCNNBackend(DetectionBackend):
    """
    MTCNN face detection backend.
    
    Specialized for face detection with landmark extraction.
    
    Settings:
        min_face_size: Minimum face size in pixels (default: 20)
        thresholds: Detection thresholds for 3 stages (default: [0.6, 0.7, 0.7])
        use_gpu: Whether to use GPU (default: True)
    """
    
    name = "mtcnn"
    
    def __init__(self, settings: Dict[str, Any] = None):
        """
        Initialize MTCNN backend.
        
        Args:
            settings: Backend configuration
        """
        self._settings = settings or {}
        self._detector = None
        self._min_face_size = self._settings.get("min_face_size", 20)
        self._thresholds = self._settings.get("thresholds", [0.6, 0.7, 0.7])
        self._use_gpu = self._settings.get("use_gpu", True)
        
        self._load_model()
    
    def _load_model(self):
        """Load MTCNN model."""
        try:
            from facenet_pytorch import MTCNN
            import torch
            
            device = torch.device("cuda" if self._use_gpu and torch.cuda.is_available() else "cpu")
            
            self._detector = MTCNN(
                min_face_size=self._min_face_size,
                thresholds=self._thresholds,
                device=device,
                keep_all=True
            )
            
            logger.info(f"MTCNN backend loaded on {device}")
        except ImportError:
            logger.error("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
            raise
        except Exception as e:
            logger.error(f"Failed to load MTCNN: {e}")
            raise
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default MTCNN settings."""
        return {
            "min_face_size": 20,
            "thresholds": [0.6, 0.7, 0.7],
            "use_gpu": True
        }
    
    async def detect(self, image_path: str, settings: Dict[str, Any] = None) -> List[Dict]:
        """
        Run MTCNN face detection.
        
        Args:
            image_path: Path to image
            settings: Optional override settings
            
        Returns:
            List of face detection dicts
        """
        if self._detector is None:
            logger.warning("MTCNN not loaded")
            return []
        
        try:
            from PIL import Image
            
            # Load image
            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size
            
            # Detect faces
            boxes, probs, landmarks = self._detector.detect(img, landmarks=True)
            
            detections = []
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Normalize bbox to 0-1
                    bbox = {
                        "x": x1 / img_w,
                        "y": y1 / img_h,
                        "w": (x2 - x1) / img_w,
                        "h": (y2 - y1) / img_h
                    }
                    
                    det = {
                        "label": "face",
                        "bbox": bbox,
                        "confidence": float(prob),
                    }
                    
                    # Add landmarks if available
                    if landmarks is not None and i < len(landmarks):
                        lm = landmarks[i]
                        det["landmarks"] = {
                            "left_eye": (float(lm[0][0]) / img_w, float(lm[0][1]) / img_h),
                            "right_eye": (float(lm[1][0]) / img_w, float(lm[1][1]) / img_h),
                            "nose": (float(lm[2][0]) / img_w, float(lm[2][1]) / img_h),
                            "mouth_left": (float(lm[3][0]) / img_w, float(lm[3][1]) / img_h),
                            "mouth_right": (float(lm[4][0]) / img_w, float(lm[4][1]) / img_h),
                        }
                    
                    detections.append(det)
            
            logger.debug(f"MTCNN detected {len(detections)} faces in {image_path}")
            return detections
            
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []
