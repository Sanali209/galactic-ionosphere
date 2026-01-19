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
import asyncio
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class GroundingDINOExtractor(AIExtractor):
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
    needs_model = True  # Requires GroundingDINO model
    
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
        if config and not isinstance(config, dict):
            # Handle ServiceLocator injecting ConfigManager
            config = {}
            
        config = config or {}
        self._class_mapping = config.get("class_mapping", self.DEFAULT_CLASS_MAPPING)
        self._box_threshold = config.get("box_threshold", 0.35)
        self._text_threshold = config.get("text_threshold", 0.25)
        self._store_as_instances = config.get("store_as_instances", True)
        
        # Build phrases from class mapping keys
        self._phrases = list(self._class_mapping.keys())
        
        logger.info(f"[GroundingDINO] Initialized with {len(self._class_mapping)} class mappings: {list(self._class_mapping.values())}")

    
    async def _ensure_model(self) -> bool:
        """Lazy load GroundingDINO model."""
        if self._model is not None:
            return self._available
            
        return await asyncio.to_thread(self._ensure_model_sync)

    def _ensure_model_sync(self) -> bool:
        """Synchronous implementation of model loading."""
        try:
            import os
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            # Determine device
            use_cuda = torch.cuda.is_available()
            self._device = "cuda" if use_cuda else "cpu"
            
            # Get HuggingFace token from environment
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            model_id = "IDEA-Research/grounding-dino-tiny"
            logger.info(f"[GroundingDINO] Loading model...")
            
            self._processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            
            
            # Strategy: Try loading, if meta tensors appear, retry with _fast_init=False
            temp_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, 
                token=hf_token,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,
            )
            
            # Check if model has meta tensors
            param = next(temp_model.parameters())
            if param.device.type == "meta":
                logger.warning("[GroundingDINO] Meta tensors detected, reloading with _fast_init=False")
                
                # Clean up and force garbage collection
                del temp_model
                import gc
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
                
                # Retry with _fast_init=False which skips meta device initialization
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_id,
                    token=hf_token,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    _fast_init=False,  # Force real tensor initialization
                )
            else:
                self._model = temp_model
            
            # Move to device
            if self._device == "cuda":
                try:
                    self._model = self._model.cuda()
                    logger.info("[GroundingDINO] Model moved to CUDA")
                except Exception as e:
                    logger.warning(f"[GroundingDINO] CUDA failed: {e}")
                    self._device = "cpu"
            
            self._model.eval()
            self._available = True
            
            # Final verification
            param = next(self._model.parameters())
            logger.info(f"[GroundingDINO] ✓ Model ready on {param.device}")
            
        except ImportError as e:
            logger.warning(f"GroundingDINO not available: {e}")
            self._available = False
        except Exception as e:
            logger.error(f"[GroundingDINO] Failed to load model: {e}")
            import traceback
            logger.debug(f"[GroundingDINO] Traceback: {traceback.format_exc()}")
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
    
    # Inherited from AIExtractor: _get_llm_service(), _get_ai_executor()
    # Note: extract() is overridden due to phrases_map requirement
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Detect objects in images.
        
        Uses LLMWorkerService for non-blocking inference when available.
        Falls back to legacy ThreadPoolExecutor method.
        """
        if not files:
            return {}
        
        logger.info(f"[GroundingDINO] Starting batch extraction for {len(files)} files")
        
        # Pre-fetch phrases (async) before offloading heavy compute
        phrases_map = {}
        for file in files:
            if self.can_process(file):
                phrases_map[file.id] = await self._get_phrases_for_file(file)
        
        # Try LLMWorkerService first
        llm_service = self._get_llm_service()
        if llm_service:
            return await self._extract_via_service(files, phrases_map, llm_service)
        
        # Fallback to legacy executor
        return await self._extract_legacy(files, phrases_map)
    
    async def _extract_via_service(self, files: List[FileRecord], phrases_map: Dict[ObjectId, List[str]], llm_service) -> Dict[ObjectId, Any]:
        """Extract detections using LLMWorkerService."""
        from src.core.llm.models import LLMJobRequest
        
        file_paths = [str(f.path) for f in files if f.id in phrases_map]
        path_to_id = {str(f.path): f.id for f in files}
        
        # Convert ObjectId keys to string for serialization
        phrases_by_path = {str(f.path): phrases_map.get(f.id, self._phrases) for f in files}
        
        request = LLMJobRequest(
            task_type="grounding_dino",
            file_paths=file_paths,
            options={
                "phrases_by_path": phrases_by_path,
                "class_mapping": self._class_mapping,
                "box_threshold": self._box_threshold,
                "text_threshold": self._text_threshold
            }
        )
        
        try:
            future = await llm_service.submit_job(request)
            result = await future
            
            if not result.success:
                logger.warning(f"GroundingDINO worker job failed: {result.error}, falling back to legacy")
                return await self._extract_legacy(files, phrases_map)
            
            # Map path-based results back to ObjectId keys
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
                logger.debug("[GroundingDINO] Worker returned empty data, falling back to legacy")
                return await self._extract_legacy(files, phrases_map)
            
            logger.info(f"[GroundingDINO] Processed {len(results)}/{len(files)} via LLMWorkerService")
            return results
            
        except Exception as e:
            logger.error(f"GroundingDINO extraction via service failed: {e}")
            return await self._extract_legacy(files, phrases_map)
    
    async def _extract_legacy(self, files: List[FileRecord], phrases_map: Dict[ObjectId, List[str]]) -> Dict[ObjectId, Any]:
        """Legacy: Extract using shared AI executor."""
        if not await self._ensure_model():
            return {}
        
        # Use inherited helper for standardized executor access
        return await self._run_in_ai_executor(self._inference_batch, files, phrases_map)
    
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """
        Required by AIExtractor but not used directly.
        
        GroundingDINO has custom extract() that handles phrases_map,
        so this is a stub that raises NotImplementedError.
        The actual worker logic is in _extract_via_llm_service().
        """
        raise NotImplementedError(
            "GroundingDINO uses custom extract() with phrases_map. "
            "See _extract_via_llm_service() for worker-based extraction."
        )

    def _inference_batch(self, files: List[FileRecord], phrases_map: Dict[ObjectId, List[str]]) -> Dict[ObjectId, Any]:
        """
        Blocking inference function to run in worker thread.
        """
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
                    
                    # Get phrases (pre-fetched)
                    phrases = phrases_map.get(file.id, list(self._phrases))
                    text_prompt = ". ".join(phrases) + "."
                    
                    # Process
                    inputs = self._processor(
                        images=image,
                        text=text_prompt,
                        return_tensors="pt"
                    ).to(self._device)
                    
                    
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                    
                    # Validate outputs before post-processing
                    if not outputs or not hasattr(outputs, 'logits'):
                        logger.warning(f"Invalid model outputs for {file.id}, skipping")
                        continue
                    
                    # Post-process (newer transformers API - no threshold params)
                    try:
                        result = self._processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            target_sizes=[image.size[::-1]]
                        )[0]
                    except (IndexError, RuntimeError) as e:
                        logger.error(f"Post-processing failed for {file.id}: {e}")
                        # This often happens with corrupted tensors or meta device issues
                        # Skip this file and continue
                        continue
                    
                    # Format detections with manual threshold filtering
                    detections = []
                    for box, score, label in zip(
                        result["boxes"],
                        result["scores"],
                        result["labels"]
                    ):
                        # Apply thresholds manually
                        if float(score) < self._box_threshold:
                            continue
                            
                        detections.append({
                            "label": label,
                            "confidence": float(score),
                            "bbox": [float(x) for x in box.tolist()],
                            "bbox_format": "xyxy"  # x1, y1, x2, y2
                        })
                    
                    results[file.id] = {
                        "detections": detections,
                        "model": "grounding-dino-tiny",
                        "phrases_used": phrases,
                        "image_size": list(image.size)
                    }
                    
                except Exception as e:
                    logger.error(f"GroundingDINO failed for {file.id}: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
        
        except Exception as e:
            logger.error(f"GroundingDINO extraction inference failed: {e}")
            
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

