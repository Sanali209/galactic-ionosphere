"""
UCoreFS - GroundingDINO Detection Extractor

Object detection using GroundingDINO for open-vocabulary detection.
Phase 3 extractor - runs one at a time for heavy processing.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class GroundingDINOExtractor(Extractor):
    """
    Open-vocabulary object detection using GroundingDINO.
    
    Features:
    - Multi-stage phrase extraction (configurable via RulesEngine)
    - Bounding box detection with confidence scores
    - Class mapping for standardized labels
    """
    
    name = "grounding_dino"
    phase = 3  # Heavy AI processing
    priority = 70
    batch_supported = False
    
    SUPPORTED_TYPES = {"image"}
    
    # Default detection phrases (can be overridden via config)
    DEFAULT_PHRASES = [
        "person", "face", "cat", "dog", "bird",
        "car", "building", "tree", "flower", "food",
        "text", "logo", "animal"
    ]
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
        
        # Config
        self._phrases = config.get("phrases", self.DEFAULT_PHRASES) if config else self.DEFAULT_PHRASES
        self._box_threshold = config.get("box_threshold", 0.35) if config else 0.35
        self._text_threshold = config.get("text_threshold", 0.25) if config else 0.25
    
    async def _ensure_model(self) -> bool:
        """Lazy load GroundingDINO model."""
        if self._model is not None:
            return self._available
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._device)
            
            self._available = True
            logger.info(f"GroundingDINO loaded on {self._device}")
            
        except ImportError:
            logger.warning("GroundingDINO not available - pip install transformers")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")
            self._available = False
        
        return self._available
    
    async def _get_phrases_for_file(self, file: FileRecord) -> List[str]:
        """
        Get detection phrases for a file.
        
        Uses RulesEngine if available for custom phrase mappings.
        """
        phrases = list(self._phrases)
        
        # Try to get custom phrases from RulesEngine
        if self.locator:
            try:
                from src.ucorefs.rules.engine import RulesEngine
                rules_engine = self.locator.get_system(RulesEngine)
                
                if hasattr(rules_engine, 'get_detection_phrases'):
                    custom = await rules_engine.get_detection_phrases(file)
                    if custom:
                        phrases = custom
            except (KeyError, ImportError):
                pass
        
        return phrases
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Detect objects in images.
        
        Args:
            files: List of image files (usually 1 for Phase 3)
            
        Returns:
            Dict mapping file_id -> detection results
        """
        if not await self._ensure_model():
            return {}
        
        results = {}
        
        try:
            import torch
            from PIL import Image
            
            for file in files:
                if not self.can_process(file):
                    continue
                
                try:
                    # Load image
                    image = Image.open(file.path).convert("RGB")
                    
                    # Get phrases for this file
                    phrases = await self._get_phrases_for_file(file)
                    text_prompt = ". ".join(phrases) + "."
                    
                    # Process
                    inputs = self._processor(
                        images=image,
                        text=text_prompt,
                        return_tensors="pt"
                    ).to(self._device)
                    
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                    
                    # Post-process
                    result = self._processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=self._box_threshold,
                        text_threshold=self._text_threshold,
                        target_sizes=[image.size[::-1]]
                    )[0]
                    
                    # Format detections
                    detections = []
                    for box, score, label in zip(
                        result["boxes"],
                        result["scores"],
                        result["labels"]
                    ):
                        detections.append({
                            "label": label,
                            "confidence": float(score),
                            "bbox": [float(x) for x in box.tolist()],
                            "bbox_format": "xyxy"  # x1, y1, x2, y2
                        })
                    
                    results[file._id] = {
                        "detections": detections,
                        "model": "grounding-dino-tiny",
                        "phrases_used": phrases,
                        "image_size": list(image.size)
                    }
                    
                except Exception as e:
                    logger.error(f"GroundingDINO failed for {file._id}: {e}")
        
        except Exception as e:
            logger.error(f"GroundingDINO extraction failed: {e}")
        
        return results
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store detection results in FileRecord."""
        try:
            detections = result.get("detections", [])
            
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            # Store in detections dict
            file.detections["grounding_dino"] = {
                "model": result.get("model", "grounding-dino-tiny"),
                "count": len(detections),
                "labels": list(set(d["label"] for d in detections)),
                "detections": detections,
                "phrases_used": result.get("phrases_used", []),
                "image_size": result.get("image_size", []),
                "created_at": datetime.now().isoformat()
            }
            
            # Update processing state
            if file.processing_state < ProcessingState.ANALYZED:
                file.processing_state = ProcessingState.ANALYZED
            
            file.last_processed_at = datetime.now()
            await file.save()
            
            logger.debug(f"Stored {len(detections)} detections for {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store detections: {e}")
            return False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
