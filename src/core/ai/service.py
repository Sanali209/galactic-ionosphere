from typing import List, Dict, Any
import numpy as np
from PIL import Image
from loguru import logger
from src.core.ai.base import AIModel, EmbeddingModel, CaptioningModel
from src.core.ai.providers import CLIPEncoder

class EmbeddingService:
    """
    Manages loading and inference of AI models.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.embedding_model: EmbeddingModel = None
        self.captioning_model: CaptioningModel = None
        
    def load_clip(self):
        """Loads the default CLIP model."""
        if self.embedding_model: return
        
        logger.info("Initializing CLIP implementation...")
        encoder = CLIPEncoder()
        encoder.load()
        self.embedding_model = encoder

    def generate_embedding(self, image_path: str) -> np.ndarray:
        if not self.embedding_model:
            self.load_clip()
            
        try:
            img = Image.open(image_path)
            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return self.embedding_model.encode_image(img)
        except Exception as e:
            logger.error(f"Error embedding {image_path}: {e}")
            return np.array([])

    def generate_text_embedding(self, text: str) -> np.ndarray:
        if not self.embedding_model:
            self.load_clip()
            
        return self.embedding_model.encode_text(text)

    async def encode_text(self, text: str) -> np.ndarray:
        """Async wrapper for generate_text_embedding to maintain compatibility.
        Returns the text embedding as a NumPy array.
        """
        return self.generate_text_embedding(text)
