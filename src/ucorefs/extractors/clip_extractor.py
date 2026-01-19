"""
UCoreFS - CLIP Embedding Extractor

Generates CLIP image embeddings for similarity search.
Phase 2 extractor - runs in batches of 20.
"""
import threading
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class CLIPExtractor(AIExtractor):
    """
    Generates CLIP embeddings for image files.
    
    Uses LLMWorkerService for non-blocking inference in separate process.
    """
    
    name = "clip"
    phase = 2
    priority = 50  # After thumbnails/metadata
    batch_supported = True
    is_cpu_heavy = True
    
    SUPPORTED_TYPES = {"image"}
    MODEL_NAME = "ViT-B/32"
    DIMENSION = 512
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Generate CLIP embeddings for images.
        
        Uses ThreadPoolExecutor for non-blocking inference (PyTorch releases GIL).
        """
        if not files:
            return {}
        
        # Filter processable files
        valid_files = [f for f in files if self.can_process(f)]
        if not valid_files:
            return {}
        
        return await self._extract_legacy(valid_files)
    
    
    async def _extract_legacy(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Legacy extraction using shared AI executor."""
        # Use inherited helper for standardized executor access
        return await self._run_in_ai_executor(self._extract_batch_sync, files)
    
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """CLIP worker not yet implemented, fallback to legacy."""
        raise NotImplementedError("CLIP worker not implemented")
    
    def _extract_batch_sync(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Synchronous batch extraction (legacy, runs in thread pool)."""
        results = {}
        try:
            import torch
            from PIL import Image
            
            # Lazy load model
            if not self._ensure_model_sync():
                return {}
            
            valid_files = []
            images = []
            
            for file in files:
                try:
                    image = Image.open(file.path).convert("RGB")
                    image_tensor = CLIPExtractor._shared_preprocess(image)
                    images.append(image_tensor)
                    valid_files.append(file)
                except Exception as e:
                    logger.error(f"Image preprocessing failed for {file._id}: {e}")
            
            if not images:
                return {}
            
            image_input = torch.stack(images).to(CLIPExtractor._shared_device)
            
            with torch.no_grad():
                image_features = CLIPExtractor._shared_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()
            
            for i, file in enumerate(valid_files):
                embedding = image_features[i].tolist()
                results[file._id] = {
                    "vector": embedding,
                    "model": self.MODEL_NAME,
                    "dimension": len(embedding)
                }
                
        except Exception as e:
            logger.error(f"CLIP legacy batch failed: {e}")
            
        return results
    
    # Class-level storage for legacy singleton model
    _shared_model = None
    _shared_preprocess = None
    _shared_device = "cpu"
    _shared_available = False
    _model_lock = threading.Lock()  # Thread-safe model loading
    
    def _ensure_model_sync(self) -> bool:
        """Legacy: Synchronous model loading for fallback. Thread-safe singleton."""
        # Quick check without lock
        if CLIPExtractor._shared_model is not None:
            return CLIPExtractor._shared_available
        
        # Acquire lock for loading
        with CLIPExtractor._model_lock:
            # Double-check after acquiring lock
            if CLIPExtractor._shared_model is not None:
                return CLIPExtractor._shared_available
            
            try:
                import torch
                import clip
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading CLIP model on {device}...")
                model, preprocess = clip.load(self.MODEL_NAME, device=device)
                
                CLIPExtractor._shared_device = device
                CLIPExtractor._shared_model = model
                CLIPExtractor._shared_preprocess = preprocess
                CLIPExtractor._shared_available = True
                
                logger.info(f"CLIP model loaded on {device} (legacy fallback)")
                
            except ImportError:
                logger.warning("CLIP not available - pip install git+https://github.com/openai/CLIP.git")
                CLIPExtractor._shared_available = False
            except AttributeError as e:
                if "has no attribute 'load'" in str(e):
                    logger.critical("Incorrect 'clip' package! Run: pip uninstall clip -y && pip install git+https://github.com/openai/CLIP.git")
                CLIPExtractor._shared_available = False
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                CLIPExtractor._shared_available = False
        
        return CLIPExtractor._shared_available
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store CLIP embedding in FileRecord."""
        import asyncio
        
        try:
            vector = result.get("vector", [])
            if not vector:
                return False
            
            file = await FileRecord.get(file_id)
            if file:
                file.embeddings["clip"] = {
                    "model": result.get("model", self.MODEL_NAME),
                    "dimension": len(vector),
                    "created_at": datetime.now().isoformat(),
                    "vector": vector
                }
                file.has_vector = True
                if file.processing_state < ProcessingState.INDEXED:
                    file.processing_state = ProcessingState.INDEXED
                await file.save()
                # Yield to UI thread after save
                await asyncio.sleep(0)
                logger.debug(f"[CLIP] Stored vector for {file_id} (dim={len(vector)})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store CLIP embedding: {e}")
            return False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
    
    async def encode_text(self, text: str) -> list:
        """Encode text query to CLIP embedding vector."""
        # Legacy implementation for text encoding (used in search)
        if not self._ensure_model_sync():
            return []
        
        try:
            import torch
            import clip
            
            text_tokens = clip.tokenize([text]).to(CLIPExtractor._shared_device)
            
            with torch.no_grad():
                embedding = CLIPExtractor._shared_model.encode_text(text_tokens)
                embedding = embedding.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return []
