from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np
from PIL import Image

class AIModel(ABC):
    """
    Abstract base class for all AI models (CLIP, BLIP, YOLO, etc.).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model (e.g. 'openai/clip-vit-base-patch32')."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of model: 'embedding', 'captioning', 'detection'."""
        pass

    @abstractmethod
    def load(self):
        """Load the model into memory/GPU."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Run inference."""
        pass

class EmbeddingModel(AIModel):
    """Specific interface for models that produce vector embeddings."""
    @abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        pass
        
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        pass

class CaptioningModel(AIModel):
    """Specific interface for models that generate text from images."""
    @abstractmethod
    def generate_caption(self, image: Image.Image) -> str:
        pass
