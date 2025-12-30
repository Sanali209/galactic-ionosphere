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
    # Class-level storage for singleton model loading
    _shared_model = None
    _shared_preprocess = None
    _shared_device = "cpu"
    _shared_available = False
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        super().__init__(locator, config)
    
    async def _ensure_model(self) -> bool:
        """Lazy load CLIP model (Singleton)."""
        if CLIPExtractor._shared_model is not None:
            return CLIPExtractor._shared_available
        
        try:
            import torch
            import clip
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(self.MODEL_NAME, device=device)
            
            CLIPExtractor._shared_device = device
            CLIPExtractor._shared_model = model
            CLIPExtractor._shared_preprocess = preprocess
            CLIPExtractor._shared_available = True
            
            logger.info(f"CLIP model loaded on {device} (Singleton)")
            
        except ImportError:
            logger.warning("CLIP not available - pip install git+https://github.com/openai/CLIP.git")
            CLIPExtractor._shared_available = False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            CLIPExtractor._shared_available = False
        
        return CLIPExtractor._shared_available
    
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
            
            # SAN-14: Offload ENTIRE batch processing to thread pool
            # This prevents model inference from blocking the main thread event loop
            results = await loop.run_in_executor(
                executor,  # Use dedicated pool if available, else default
                self._extract_batch_sync,
                files
            )
            
        except Exception as e:
            logger.error(f"CLIP extraction batch failed: {e}")
        
        return results

    def _extract_batch_sync(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Synchronous batch extraction (runs in thread pool).
        
        Performs both preprocessing and inference off main thread.
        """
        results = {}
        try:
            import torch
            from PIL import Image
            
            # Prepare batch
            valid_files = []
            images = []
            
            for file in files:
                if not self.can_process(file):
                    continue
                    
                try:
                    image = Image.open(file.path).convert("RGB")
                    image_tensor = CLIPExtractor._shared_preprocess(image)
                    images.append(image_tensor)
                    valid_files.append(file)
                except Exception as e:
                    logger.error(f"Image preprocessing failed for {file._id}: {e}")
            
            if not images:
                return {}
            
            # Stack images
            image_input = torch.stack(images).to(CLIPExtractor._shared_device)
            
            # Inference
            with torch.no_grad():
                image_features = CLIPExtractor._shared_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()
            
            # Map results
            for i, file in enumerate(valid_files):
                embedding = image_features[i].tolist()
                results[file._id] = {
                    "vector": embedding,
                    "model": self.MODEL_NAME,
                    "dimension": len(embedding)
                }
                
        except Exception as e:
            logger.error(f"CLIP synchronous batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        return results
    

    
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
            text_tokens = clip.tokenize([text]).to(CLIPExtractor._shared_device)
            
            with torch.no_grad():
                embedding = CLIPExtractor._shared_model.encode_text(text_tokens)
                embedding = embedding.cpu().numpy().flatten().tolist()
            
            logger.debug(f"CLIP text encoding: '{text[:50]}...' -> dim={len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return []
