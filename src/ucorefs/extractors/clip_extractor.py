"""
UCoreFS - CLIP Embedding Extractor

Generates CLIP image embeddings for similarity search.
Phase 2 extractor - runs in batches of 20.
"""
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class CLIPExtractor(Extractor):
    """
    Generates CLIP embeddings for image files.
    
    Uses OpenAI's CLIP model (or compatible) to create
    embeddings for visual similarity search.
    """
    
    name = "clip"
    phase = 2
    priority = 50  # After thumbnails/metadata
    batch_supported = True
    is_cpu_heavy = True  # SAN-14: PIL image preprocessing operations
    
    # Supported file types
    SUPPORTED_TYPES = {"image"}
    
    # CLIP configuration
    MODEL_NAME = "ViT-B/32"
    DIMENSION = 512
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self._available = False
    
    async def _ensure_model(self) -> bool:
        """Lazy load CLIP model."""
        if self._model is not None:
            return self._available
        
        try:
            import torch
            import clip
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = clip.load(self.MODEL_NAME, device=self._device)
            self._available = True
            logger.info(f"CLIP model loaded on {self._device}")
            
        except ImportError:
            logger.warning("CLIP not available - pip install git+https://github.com/openai/CLIP.git")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self._available = False
        
        return self._available
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Generate CLIP embeddings for images.
        
        Args:
            files: List of image files
            
        Returns:
            Dict mapping file_id -> embedding vector
        """
        if not await self._ensure_model():
            return {}
        
        results = {}
        
        try:
            import torch
            import asyncio
            
            # SAN-14 Phase 2: Get dedicated AI executor if available
            loop = asyncio.get_event_loop()
            executor = None
            
            if self.locator:
                try:
                    from src.ucorefs.processing.pipeline import ProcessingPipeline
                    pipeline = self.locator.get_system(ProcessingPipeline)
                    executor = pipeline.get_ai_executor()
                    if executor:
                        logger.debug(f"Using dedicated AI thread pool for {len(files)} images")
                except (KeyError, AttributeError):
                    pass  # Fall back to default pool
            
            for file in files:
                if not self.can_process(file):
                    continue
                
                try:
                    # SAN-14: Offload PIL preprocessing to thread pool (dedicated or default)
                    image_tensor = await loop.run_in_executor(
                        executor,  # Use dedicated pool if available, else default
                        self._preprocess_image_sync,
                        file.path
                    )
                    
                    if image_tensor is None:
                        continue
                    
                    # Move to device and generate embedding (GPU releases GIL)
                    image_input = image_tensor.unsqueeze(0).to(self._device)
                    
                    with torch.no_grad():
                        embedding = self._model.encode_image(image_input)
                        embedding = embedding.cpu().numpy().flatten().tolist()
                    
                    results[file._id] = {
                        "vector": embedding,
                        "model": self.MODEL_NAME,
                        "dimension": len(embedding)
                    }
                    
                except Exception as e:
                    logger.error(f"CLIP embedding failed for {file._id}: {e}")
        
        except Exception as e:
            logger.error(f"CLIP extraction batch failed: {e}")
        
        return results
    
    def _preprocess_image_sync(self, image_path: str):
        """
        Synchronous image preprocessing (runs in thread pool).
        
        This method offloads PIL operations to prevent event loop blocking.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor or None on error
        """
        try:
            from PIL import Image
            
            # PIL operations run in thread pool
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._preprocess(image)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed for {image_path}: {e}")
            return None
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store CLIP embedding in MongoDB + FAISS."""
        try:
            from src.ucorefs.vectors.models import EmbeddingRecord
            
            vector = result.get("vector", [])
            if not vector:
                return False
            
            # Upsert embedding record
            existing = await EmbeddingRecord.find_one({
                "file_id": file_id,
                "provider": "clip"
            })
            
            if existing:
                existing.vector = vector
                existing.dimension = len(vector)
                existing.model_version = result.get("model", self.MODEL_NAME)
                existing.updated_at = datetime.now()
                await existing.save()
            else:
                record = EmbeddingRecord(
                    file_id=file_id,
                    provider="clip",
                    vector=vector,
                    dimension=len(vector),
                    model_version=result.get("model", self.MODEL_NAME)
                )
                await record.save()
            
            # Update FileRecord
            file = await FileRecord.get(file_id)
            if file:
                file.embeddings["clip"] = {
                    "model": result.get("model", self.MODEL_NAME),
                    "dimension": len(vector),
                    "created_at": datetime.now().isoformat()
                }
                file.has_vector = True
                if file.processing_state < ProcessingState.INDEXED:
                    file.processing_state = ProcessingState.INDEXED
                await file.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store CLIP embedding: {e}")
            return False
    
    def can_process(self, file: FileRecord) -> bool:
        """Only process image files."""
        return file.file_type in self.SUPPORTED_TYPES
    
    async def encode_text(self, text: str) -> list:
        """
        Encode text query to CLIP embedding vector.
        
        Args:
            text: Text query to encode
            
        Returns:
            List of floats representing the embedding vector
        """
        if not await self._ensure_model():
            return []
        
        try:
            import torch
            import clip
            
            # Tokenize and encode text
            text_tokens = clip.tokenize([text]).to(self._device)
            
            with torch.no_grad():
                embedding = self._model.encode_text(text_tokens)
                embedding = embedding.cpu().numpy().flatten().tolist()
            
            logger.debug(f"CLIP text encoding: '{text[:50]}...' -> dim={len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return []
