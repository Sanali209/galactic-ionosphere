from typing import List, Union, Any
from loguru import logger
import asyncio

# Import localized to avoid crash if dependencies missing during early load
try:
    from sentence_transformers import SentenceTransformer
    from PIL import Image
except ImportError:
    SentenceTransformer = None
    Image = None

class EmbeddingService:
    """
    Wrapper for Local AI Models (CLIP/ViT).
    """
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self.model = None

    async def load(self):
        """Loads model (potentially slow, run in thread if needed)."""
        if not SentenceTransformer:
            logger.warning("sentence-transformers not installed. Embedding disabled.")
            return

        if self.model:
            return

        logger.info(f"Loading AI Model: {self.model_name}...")
        try:
            # Running in thread to avoid blocking loop during heavy load
            self.model = await asyncio.to_thread(SentenceTransformer, self.model_name)
            logger.info("AI Model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")

    async def encode_text(self, text: str) -> List[float]:
        if not self.model:
            await self.load()
            if not self.model: return []

        try:
            # encode is synchronous in sentence_transformers
            return await asyncio.to_thread(lambda: self.model.encode(text).tolist())
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

    async def encode_image(self, image_path: str) -> List[float]:
        if not self.model or not Image:
            await self.load()
            if not self.model: return []
            
        try:
            # Image.open is fast, but encode is slow
            img = Image.open(image_path)
            return await asyncio.to_thread(lambda: self.model.encode(img).tolist())
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return []
