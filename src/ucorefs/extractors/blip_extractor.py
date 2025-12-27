"""
UCoreFS - BLIP Caption Extractor

Generates image captions and text embeddings using BLIP.
Phase 3 extractor - runs one at a time for heavy processing.
"""
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class BLIPExtractor(Extractor):
    """
    Generates captions and embeddings using BLIP.
    
    Produces:
    - Image caption (natural language description)
    - Text embedding for semantic search
    """
    
    name = "blip"
    phase = 3  # Heavy processing - Phase 3
    priority = 80
    batch_supported = False  # Process one at a time
    
    SUPPORTED_TYPES = {"image"}
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._available = False
    
    async def _ensure_model(self) -> bool:
        """Lazy load BLIP model."""
        if self._model is not None:
            return self._available
        
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_name = "Salesforce/blip-image-captioning-base"
            self._processor = BlipProcessor.from_pretrained(model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(model_name).to(self._device)
            
            self._available = True
            logger.info(f"BLIP model loaded on {self._device}")
            
        except ImportError:
            logger.warning("BLIP not available - pip install transformers")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            self._available = False
        
        return self._available
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Generate captions for images.
        
        Args:
            files: List of image files (usually 1 for Phase 3)
            
        Returns:
            Dict mapping file_id -> caption result
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
                    
                    # Generate caption
                    inputs = self._processor(image, return_tensors="pt").to(self._device)
                    
                    with torch.no_grad():
                        output = self._model.generate(**inputs, max_length=50)
                        caption = self._processor.decode(output[0], skip_special_tokens=True)
                    
                    results[file._id] = {
                        "caption": caption,
                        "model": "blip-base"
                    }
                    
                except Exception as e:
                    logger.error(f"BLIP caption failed for {file._id}: {e}")
        
        except Exception as e:
            logger.error(f"BLIP extraction failed: {e}")
        
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
            
            # Store caption
            file.ai_caption = caption
            
            # Update processing state
            if file.processing_state < ProcessingState.ANALYZED:
                file.processing_state = ProcessingState.ANALYZED
            
            file.last_processed_at = datetime.now()
            await file.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store BLIP caption: {e}")
            return False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
