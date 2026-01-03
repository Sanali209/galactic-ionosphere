"""
UCoreFS - GroundingDINO Detection Extractor

Object detection using GroundingDINO for open-vocabulary detection.
Phase 3 extractor - runs one at a time for heavy processing.

Configurable Ontology:
    The class_mapping config defines what to detect and how to store it.
    Format: {"search_phrase": "class_name"}
    
    Example:
        class_mapping = {
            "person": "person",
            "face": "face",
            "human face": "face",  # Alternative phrase -> same class
            "cat": "animal",
            "dog": "animal",
        }
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
    - Configurable class mapping (ontology) for detection phrases
    - Bounding box detection with confidence scores
    - Storage as DetectionInstance records with relations
    
    Configuration:
        class_mapping: Dict[str, str] - Maps search phrases to class names
            {"person": "person", "face": "face", "human face": "face"}
        box_threshold: float - Min confidence for bbox (default: 0.35)
        text_threshold: float - Min confidence for text match (default: 0.25)
        store_as_instances: bool - Store as DetectionInstance vs dict (default: True)
    
    Usage:
        # Configure in settings.yaml:
        processing:
          grounding_dino:
            class_mapping:
              person: person
              face: face
              cat: animal
              dog: animal
    """
    
    name = "grounding_dino"
    phase = 3  # Heavy AI processing
    priority = 70
    batch_supported = False
    
    SUPPORTED_TYPES = {"image"}
    
    # Default class mapping ontology
    DEFAULT_CLASS_MAPPING = {
        "person": "person",
        "face": "face", 
        "human face": "face",
        "cat": "cat",
        "dog": "dog",
        "bird": "bird",
        "car": "vehicle",
        "truck": "vehicle",
        "motorcycle": "vehicle",
        "building": "architecture",
        "text": "text",
        "logo": "logo",
    }
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
        
        # Config
        config = config or {}
        self._class_mapping = config.get("class_mapping", self.DEFAULT_CLASS_MAPPING)
        self._box_threshold = config.get("box_threshold", 0.35)
        self._text_threshold = config.get("text_threshold", 0.25)
        self._store_as_instances = config.get("store_as_instances", True)
        
        # Build phrases from class mapping keys
        self._phrases = list(self._class_mapping.keys())

    
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
            import os
            logger.error(f"Failed to load GroundingDINO: {e}")
            if "401" in str(e) or "403" in str(e) or "not found" in str(e).lower():
                if not os.environ.get("HF_TOKEN"):
                    logger.critical("Hugging Face token missing! Create a .env file with HF_TOKEN=your_token")
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
        """
        Store detection results.
        
        If store_as_instances is True (default), creates DetectionInstance records.
        Otherwise stores in FileRecord.detections dict.
        """
        try:
            detections = result.get("detections", [])
            
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            image_size = result.get("image_size", [1, 1])
            img_w, img_h = image_size if len(image_size) >= 2 else (1, 1)
            
            if self._store_as_instances:
                # Store as DetectionInstance records
                await self._store_as_detection_instances(
                    file_id=file_id,
                    detections=detections,
                    img_w=img_w,
                    img_h=img_h
                )
            else:
                # Legacy: store in FileRecord.detections dict
                file.detections["grounding_dino"] = {
                    "model": result.get("model", "grounding-dino-tiny"),
                    "count": len(detections),
                    "labels": list(set(d["label"] for d in detections)),
                    "detections": detections,
                    "phrases_used": result.get("phrases_used", []),
                    "image_size": image_size,
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
    
    async def _store_as_detection_instances(
        self,
        file_id: ObjectId,
        detections: List[dict],
        img_w: int,
        img_h: int
    ) -> List[ObjectId]:
        """
        Store detections as DetectionInstance records with DetectionClass.
        
        Args:
            file_id: Parent file ID
            detections: List of detection dicts from extract()
            img_w, img_h: Image dimensions for normalization
            
        Returns:
            List of created DetectionInstance IDs
        """
        from src.ucorefs.detection.models import DetectionInstance, DetectionClass
        
        instance_ids = []
        
        for i, det in enumerate(detections):
            # Get class name from mapping
            raw_label = det.get("label", "unknown")
            class_name = self._class_mapping.get(raw_label, raw_label)
            
            # Get or create DetectionClass
            detection_class = await self._get_or_create_class(class_name)
            
            # Convert bbox from xyxy to normalized xywh
            bbox = det.get("bbox", [0, 0, 0, 0])
            if det.get("bbox_format") == "xyxy":
                x1, y1, x2, y2 = bbox
                norm_bbox = {
                    "x": x1 / img_w if img_w > 0 else 0,
                    "y": y1 / img_h if img_h > 0 else 0,
                    "w": (x2 - x1) / img_w if img_w > 0 else 0,
                    "h": (y2 - y1) / img_h if img_h > 0 else 0,
                }
            else:
                norm_bbox = {"x": 0, "y": 0, "w": 0, "h": 0}
            
            # Create DetectionInstance
            instance = DetectionInstance(
                name=f"{class_name}_{i}",
                file_id=file_id,
                detection_class_id=detection_class.id if detection_class else None,
                bbox=norm_bbox,
                confidence=det.get("confidence", 0.0),
                is_virtual=True,
                driver_type="detection"
            )
            await instance.save()
            instance_ids.append(instance.id)
        
        return instance_ids
    
    async def _get_or_create_class(self, class_name: str):
        """
        Get or create DetectionClass for the given class name.
        
        Args:
            class_name: Normalized class name (e.g., "person", "face", "vehicle")
            
        Returns:
            DetectionClass record
        """
        from src.ucorefs.detection.models import DetectionClass
        
        # Search for existing class
        existing = await DetectionClass.find_one({"class_name": class_name})
        if existing:
            return existing
        
        # Create new class
        new_class = DetectionClass(
            name=class_name,
            class_name=class_name
        )
        await new_class.save()
        
        logger.debug(f"Created DetectionClass: {class_name}")
        return new_class
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
    
    def get_class_mapping(self) -> Dict[str, str]:
        """
        Get the current class mapping ontology.
        
        Returns:
            Dict mapping search phrases to class names.
            
        Example:
            {"person": "person", "face": "face", "human face": "face"}
        """
        return dict(self._class_mapping)
    
    def set_class_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Update the class mapping ontology.
        
        Args:
            mapping: Dict mapping search phrases to class names.
        """
        self._class_mapping = mapping
        self._phrases = list(mapping.keys())
        logger.info(f"Updated GroundingDINO class mapping: {len(mapping)} entries")

