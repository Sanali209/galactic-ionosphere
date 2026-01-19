"""
UCoreFS - WD-Tagger Service

Foundation System for AI-powered image tagging using WD-Tagger models.
Model is loaded at startup and shared across all tagging requests.

Usage:
    service = locator.get_system(WDTaggerService)
    result = await service.tag_image(Path("/path/to/image.jpg"))
"""
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

from loguru import logger

from src.core.base_system import BaseSystem


class WDTaggerService(BaseSystem):
    """
    WD-Tagger AI tagging service.
    
    Implements BaseSystem for Foundation integration:
    - Singleton via ServiceLocator
    - Model loaded at startup (initialize)
    - Thread-safe inference via tag_image()
    
    Configuration (via config):
        wd_tagger.enabled: bool - Enable/disable service
        wd_tagger.model_repo: str - HuggingFace model repo
        wd_tagger.general_threshold: float - Tag confidence threshold
        wd_tagger.character_threshold: float - Character tag threshold
    """
    
    # No dependencies - loads model independently
    depends_on = []
    
    # Model configuration defaults
    DEFAULT_MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
    DEFAULT_GENERAL_THRESHOLD = 0.35
    DEFAULT_CHARACTER_THRESHOLD = 0.75
    
    # Supported image extensions
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
    
    def __init__(self, locator, config=None):
        """Initialize WDTaggerService."""
        super().__init__(locator, config)
        
        # Model state (loaded in initialize)
        self._model = None
        self._transform = None
        self._labels = None
        self._is_ready = False  # Track if model is loaded and ready
        
        # Config - use defaults if wd_tagger section not in config
        self.enabled = True
        self.model_repo = self.DEFAULT_MODEL_REPO
        self.general_threshold = self.DEFAULT_GENERAL_THRESHOLD
        self.character_threshold = self.DEFAULT_CHARACTER_THRESHOLD
        
        # Try to read from config if available
        if self.config:
            try:
                self.enabled = self.config.get("wd_tagger", "enabled")
            except (AttributeError, ValueError):
                pass
            try:
                self.model_repo = self.config.get("wd_tagger", "model_repo")
            except (AttributeError, ValueError):
                pass
            try:
                self.general_threshold = self.config.get("wd_tagger", "general_threshold")
            except (AttributeError, ValueError):
                pass
            try:
                self.character_threshold = self.config.get("wd_tagger", "character_threshold")
            except (AttributeError, ValueError):
                pass
    
    async def initialize(self) -> None:
        """Load model at startup (called by ServiceLocator)."""
        print(">>> WDTaggerService.initialize() CALLED")  # Debug
        
        if not self.enabled:
            logger.info("WDTaggerService disabled by config")
            await super().initialize()
            return
        
        logger.info("WDTaggerService initializing...")
        print(">>> WDTaggerService: Starting model load...")  # Debug
        
        try:
            # Load model in thread to not block event loop
            await asyncio.to_thread(self._load_model_sync)
            print(">>> WDTaggerService: Model loaded successfully")  # Debug
            logger.info("WDTaggerService ready")
        except Exception as e:
            import traceback
            print(f">>> WDTaggerService: FAILED - {e}")  # Debug
            logger.error(f"WDTaggerService failed to initialize: {e}")
            logger.error(traceback.format_exc())
            # Don't raise - allow app to start without tagging
            self._is_ready = False
        else:
            # Model loaded successfully
            self._is_ready = True
        
        await super().initialize()
        logger.info(f"WDTaggerService ready: {self.is_ready}")
    
    async def shutdown(self) -> None:
        """Cleanup model resources."""
        logger.info("WDTaggerService shutting down")
        
        self._is_ready = False  # Mark as not ready
        
        # Release model (helps with GPU memory)
        self._model = None
        self._transform = None
        self._labels = None
        
        await super().shutdown()
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready to process images."""
        return self._is_ready and self._model is not None
    
    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread)."""
        import torch
        import timm
        from timm.data import create_transform, resolve_data_config
        from huggingface_hub import hf_hub_download
        import pandas as pd
        import numpy as np
        
        logger.info(f"Loading WD-Tagger model: {self.model_repo}")
        
        # Load model
        model = timm.create_model(f"hf-hub:{self.model_repo}").eval()
        state_dict = timm.models.load_state_dict_from_hf(self.model_repo)
        model.load_state_dict(state_dict)
        
        # Load labels
        csv_path = hf_hub_download(
            repo_id=self.model_repo,
            filename="selected_tags.csv"
        )
        df = pd.read_csv(csv_path, usecols=["name", "category"])
        labels = {
            "names": df["name"].tolist(),
            "rating": list(np.where(df["category"] == 9)[0]),
            "general": list(np.where(df["category"] == 0)[0]),
            "character": list(np.where(df["category"] == 4)[0]),
        }
        
        # Create transform
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("WD-Tagger using CUDA")
        
        # Store
        self._model = model
        self._labels = labels
        self._transform = transform
    
    async def tag_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Tag a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with tags, characters, rating, scores
            None if service not ready or error
        """
        # Comprehensive readiness check for graceful degradation during startup/shutdown
        if not self.is_ready or self._model is None or self._transform is None:
            logger.debug(f"WDTaggerService not ready, skipping {image_path.name}")
            return None
        
        try:
            return await asyncio.to_thread(self._inference_sync, image_path)
        except Exception as e:
            logger.warning(f"WDTaggerService.tag_image failed: {e}")  # Warning, not error
            return None
    
    def _inference_sync(self, image_path: Path) -> Dict[str, Any]:
        """Synchronous inference (runs in thread)."""
        import torch
        from torch.nn import functional as F
        from PIL import Image
        
        # Load and preprocess
        img = Image.open(image_path)
        img = self._ensure_rgb(img)
        img = self._pad_square(img)
        
        # Transform
        inputs = self._transform(img).unsqueeze(0)
        inputs = inputs[:, [2, 1, 0]]  # RGB to BGR
        
        # Move to GPU if model is on GPU
        if next(self._model.parameters()).is_cuda:
            inputs = inputs.cuda()
        
        # Inference
        with torch.inference_mode():
            outputs = self._model.forward(inputs)
            outputs = F.sigmoid(outputs)
            probs = outputs.squeeze(0).cpu().numpy()
        
        # Parse results
        return self._parse_tags(probs)
    
    def _parse_tags(self, probs) -> Dict[str, Any]:
        """Parse model output into tags."""
        import numpy as np
        
        labels = self._labels
        
        # Rating (highest confidence)
        rating_probs = [(labels["names"][i], probs[i]) for i in labels["rating"]]
        rating = max(rating_probs, key=lambda x: x[1])[0]
        
        # General tags above threshold
        general_tags = []
        for i in labels["general"]:
            if probs[i] > self.general_threshold:
                general_tags.append((labels["names"][i], float(probs[i])))
        general_tags.sort(key=lambda x: x[1], reverse=True)
        
        # Character tags above threshold
        character_tags = []
        for i in labels["character"]:
            if probs[i] > self.character_threshold:
                character_tags.append((labels["names"][i], float(probs[i])))
        character_tags.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "tags": [t[0] for t in general_tags],
            "characters": [c[0] for c in character_tags],
            "rating": rating,
            "scores": {
                **{t[0]: t[1] for t in general_tags[:20]},
                **{c[0]: c[1] for c in character_tags},
            }
        }
    
    @staticmethod
    def _ensure_rgb(image) -> 'Image':
        """Ensure image is RGB."""
        from PIL import Image
        
        if image.mode not in ["RGB", "RGBA"]:
            image = (
                image.convert("RGBA") if "transparency" in image.info
                else image.convert("RGB")
            )
        
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        
        return image
    
    @staticmethod
    def _pad_square(image) -> 'Image':
        """Pad image to square with white background."""
        from PIL import Image
        
        w, h = image.size
        px = max(w, h)
        canvas = Image.new("RGB", (px, px), (255, 255, 255))
        canvas.paste(image, ((px - w) // 2, (px - h) // 2))
        return canvas
    
    def can_process_extension(self, ext: str) -> bool:
        """Check if extension is supported."""
        return ext.lower().lstrip(".") in self.IMAGE_EXTENSIONS
