"""
UCoreFS - BLIP Caption Extractor

Generates image captions and text embeddings using BLIP.
Phase 3 extractor - runs one at a time for heavy processing.

Refactored to use AIExtractor for non-blocking inference.
"""
import threading
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class BLIPExtractor(AIExtractor):
    """
    Generates captions and embeddings using BLIP.
    
    Uses AIExtractor's try-worker/fallback pattern.
    """
    
    name = "blip"
    phase = 3
    priority = 80
    batch_supported = False
    needs_model = True  # Requires BLIP model
    
    SUPPORTED_TYPES = {"image"}
    
    # Class-level lock for thread-safe model loading
    _model_lock = threading.Lock()
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._llm_service = None
        # Legacy model storage
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
    
    # Inherited from AIExtractor: _get_llm_service(), extract()
    
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """Extract using LLMWorkerService (not yet implemented for BLIP)."""
        # BLIP doesn't have LLM worker implementation yet, fall through to legacy
        raise NotImplementedError("BLIP worker not implemented")

    
    async def _extract_legacy(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Legacy extraction using shared AI executor."""
        if not await self._ensure_model():
            return {}
        
        # Use inherited helper for standardized executor access
        return await self._run_in_ai_executor(self._inference_batch, files)
    
    async def _ensure_model(self) -> bool:
        """Lazy load BLIP model (legacy)."""
        if self._model is not None:
            return self._available
        
        import asyncio
        return await asyncio.to_thread(self._ensure_model_sync)

    def _ensure_model_sync(self) -> bool:
        """Synchronous model loading with thread safety."""
        # Fast path: already loaded
        if self._model is not None:
            return self._available
        
        # Thread-safe double-check locking
        with BLIPExtractor._model_lock:
            # Re-check after acquiring lock
            if self._model is not None:
                return self._available
            
            try:
                import os
                import torch
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                # Force CPU to avoid meta tensor issues
                self._device = "cpu"
                hf_token = os.environ.get("HF_TOKEN")
                
                model_name = "Salesforce/blip-image-captioning-base"
                logger.info(f"Loading BLIP model on {self._device}...")
                
                self._processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    model_name, 
                    token=hf_token,
                    torch_dtype=torch.float32  # Avoid half-precision issues
                )
                self._model = self._model.to(self._device)
                self._model.eval()  # Set to eval mode
                
                self._available = True
                logger.info(f"BLIP model loaded on {self._device}")
                
            except ImportError:
                logger.warning("BLIP not available - pip install transformers")
                self._available = False
            except Exception as e:
                logger.error(f"Failed to load BLIP model: {e}")
                self._available = False
        
        return self._available
    
    def _inference_batch(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Blocking inference (legacy)."""
        results = {}
        try:
            import torch
            from PIL import Image
            
            for file in files:
                try:
                    image = Image.open(file.path).convert("RGB")
                    inputs = self._processor(image, return_tensors="pt").to(self._device)
                    
                    with torch.no_grad():
                        output = self._model.generate(**inputs, max_length=50)
                        caption = self._processor.decode(output[0], skip_special_tokens=True)
                    
                    results[file._id] = {"caption": caption, "model": "blip-base"}
                    
                except Exception as e:
                    logger.error(f"BLIP caption failed for {file._id}: {e}")
        
        except Exception as e:
            logger.error(f"BLIP legacy batch failed: {e}")
            
        return results
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store BLIP caption in FileRecord."""
        try:
            caption = result.get("caption", "")
            if not caption:
                return False
            
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            file.ai_caption = caption
            logger.info(f"Stored BLIP caption for {file.name}: '{caption[:50]}...'")
            
            # Auto-fill description if empty
            auto_fill = True
            if self.locator and hasattr(self.locator, 'config'):
                try:
                    meta_config = self.locator.config.data.metadata
                    auto_fill = getattr(meta_config, 'auto_fill_description_from_blip', True)
                except Exception:
                    pass
            
            if auto_fill and not file.description:
                file.description = caption
            
            if file.processing_state < ProcessingState.ANALYZED:
                file.processing_state = ProcessingState.ANALYZED
            
            file.last_processed_at = datetime.now()
            await file.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store BLIP caption: {e}")
            return False
