"""
UCoreFS - DINO Extractor

Extracts DINO embeddings for image files using ViT-Small model.
Ported from SLM/vision/imagetotensor/backends/DINO.py.
"""
from typing import List, Dict, Any, Optional
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord


class DINOExtractor(Extractor):
    """
    DINO embedding extractor for images.
    
    Uses Facebook's DINO (self-DIstillation with NO labels) 
    ViT-Small model via timm for visual embeddings.
    
    Output: 384-dimensional feature vector stored in FileRecord.embeddings
    
    Settings:
        model: Model variant (default: vit_small_patch16_224.dino)
        use_gpu: Whether to use GPU (default: True)
        
    Note: Runs in Phase 2 as batch processing is efficient.
    """
    
    name = "dino"
    phase = 2
    priority = 10  # After metadata/thumbnails
    batch_supported = True
    
    VECTOR_SIZE = 384
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        """
        Initialize DINO extractor.
        
        Args:
            locator: ServiceLocator
            config: Configuration with 'model' and 'use_gpu' keys
        """
        super().__init__(locator, config)
        self._model = None
        self._transform = None
        self._device = None
        
        # Config
        self._model_name = self.config.get("model", "vit_small_patch16_224.dino")
        self._use_gpu = self.config.get("use_gpu", True)
    
    def _ensure_model_loaded(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        
        try:
            import timm
            import torch
            from torchvision import transforms
            
            self._device = torch.device(
                "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            )
            
            logger.info(f"Loading DINO model: {self._model_name} on {self._device}")
            
            self._model = timm.create_model(self._model_name, pretrained=True)
            self._model = self._model.to(self._device)
            self._model.eval()
            
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            logger.info("DINO model loaded successfully")
            
        except ImportError as e:
            logger.error(f"DINO dependencies missing: {e}")
            logger.error("Install with: pip install timm torch torchvision")
            raise
    
    def can_process(self, file: FileRecord) -> bool:
        """Check if file is a supported image."""
        if not file.extension:
            return False
        return file.extension.lower() in self.IMAGE_EXTENSIONS
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Extract DINO embeddings from images.
        
        Args:
            files: List of FileRecord objects
            
        Returns:
            Dict mapping file_id -> embedding ndarray
        """
        results = {}
        
        try:
            self._ensure_model_loaded()
        except Exception:
            return results
        
        import torch
        from PIL import Image
        
        for file in files:
            try:
                # Load and preprocess image
                img = Image.open(file.path).convert("RGB")
                img_tensor = self._transform(img).unsqueeze(0).to(self._device)
                
                # Extract features
                with torch.no_grad():
                    features = self._model.forward_features(img_tensor)
                    output = self._model.forward_head(features, pre_logits=True)
                    embedding = output.squeeze().cpu().numpy().flatten()
                
                results[file._id] = embedding
                logger.debug(f"DINO extracted embedding for {file.name}")
                
            except Exception as e:
                logger.error(f"DINO extraction failed for {file.path}: {e}")
                results[file._id] = None
        
        return results
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """
        Store DINO embedding in FileRecord.
        
        Args:
            file_id: File ObjectId
            result: Embedding ndarray
            
        Returns:
            True if stored successfully
        """
        if result is None:
            return False
        
        try:
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            # Store in embeddings dict
            if not hasattr(file, 'embeddings') or file.embeddings is None:
                file.embeddings = {}
            
            file.embeddings['dino'] = result.tolist()  # Convert to list for JSON storage
            await file.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store DINO embedding for {file_id}: {e}")
            return False
