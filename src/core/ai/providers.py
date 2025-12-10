from typing import Any, List
import numpy as np
from PIL import Image
from loguru import logger
from src.core.ai.base import EmbeddingModel, CaptioningModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class CLIPEncoder(EmbeddingModel):
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self._model_name = model_name
        self._model = None
        
    @property
    def name(self) -> str:
        return self._model_name
        
    @property
    def type(self) -> str:
        return "embedding"

    def load(self):
        if not SentenceTransformer:
            logger.error("sentence-transformers not installed. Cannot load CLIP.")
            return

        logger.info(f"Loading CLIP model: {self._model_name}...")
        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")

    def predict(self, data: Any) -> Any:
        # Generic predict, maybe not used directly if we have specific encode methods
        pass

    def encode_image(self, image: Image.Image) -> np.ndarray:
        if not self._model:
            self.load()
            if not self._model: return np.array([])
            
        # SentenceTransformer CLIP supports raw images
        return self._model.encode(image)

    def encode_text(self, text: str) -> np.ndarray:
        if not self._model:
            self.load()
            if not self._model: return np.array([])
            
        return self._model.encode(text)
