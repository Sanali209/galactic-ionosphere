"""
UCoreFS - WD-Tagger Extractor

Auto-tagging extractor using SmilingWolf's WD-Tagger models.
Generates hierarchical tags for images (general, character, rating).

Part of Phase 2 processing pipeline.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.models.file_record import FileRecord


class WDTaggerExtractor(Extractor):
    """
    WD-Tagger auto-tagging extractor.
    
    Delegates to WDTaggerService for AI inference.
    The service is a Foundation System that loads the model at startup.
    
    Tag hierarchy:
    - auto/wd_tag/{tag} - General content tags
    - auto/wd_character/{character} - Character names
    - auto/wd_rating/{rating} - Content rating (general, sensitive, etc.)
    
    Configuration:
        enabled: bool - Whether to enable this extractor
    """
    
    name: str = "wd_tagger"
    phase: int = 2
    priority: int = 50  # Run after thumbnails, before embeddings
    batch_supported: bool = False  # Process one at a time (GPU memory)
    is_cpu_heavy: bool = True  # SAN-14: AI inference with image preprocessing (PIL operations)
    
    # Model configuration (passed to service via config)
    DEFAULT_MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
    DEFAULT_GENERAL_THRESHOLD = 0.35
    DEFAULT_CHARACTER_THRESHOLD = 0.75
    
    # Image extensions to process
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
    
    def __init__(self, locator=None, config: Dict[str, Any] = None):
        """Initialize WD-Tagger extractor."""
        super().__init__(locator, config)
        
        # Config
        self.enabled = self.config.get("enabled", True)
        self.model_repo = self.config.get("model_repo", self.DEFAULT_MODEL_REPO)
        self.general_threshold = self.config.get(
            "general_threshold", self.DEFAULT_GENERAL_THRESHOLD
        )
        self.character_threshold = self.config.get(
            "character_threshold", self.DEFAULT_CHARACTER_THRESHOLD
        )
    
    def can_process(self, file: FileRecord) -> bool:
        """Check if file is a processable image."""
        if not self.enabled:
            return False
        
        # Check if service is ready (don't queue files if model still loading)
        service = self._get_tagger_service()
        if not service or not service.is_ready:
            return False
        
        ext = (file.extension or "").lower().lstrip(".")
        return ext in self.IMAGE_EXTENSIONS
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Extract tags from images using WDTaggerService.
        
        Returns:
            Dict mapping file_id -> {
                "tags": List[str],
                "characters": List[str],
                "rating": str,
                "scores": Dict[str, float]
            }
        """
        results = {}
        
        # Get service from locator
        service = self._get_tagger_service()
        if not service:
            logger.warning("WDTaggerService not available, skipping tagging")
            return {f._id: None for f in files}
        
        for file in files:
            try:
                tags_result = await service.tag_image(Path(file.path))
                results[file._id] = tags_result
                
            except Exception as e:
                logger.error(f"WD-Tagger failed for {file.path}: {e}")
                results[file._id] = None
        
        return results
    
    def _get_tagger_service(self):
        """Get WDTaggerService from locator."""
        if not self.locator:
            return None
        try:
            from src.ucorefs.services.wd_tagger_service import WDTaggerService
            return self.locator.get_system(WDTaggerService)
        except (KeyError, ImportError):
            return None
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """Store extracted tags in database via TagManager."""
        if result is None:
            return False
        
        try:
            # Get TagManager from locator
            if self.locator:
                from src.ucorefs.tags.manager import TagManager
                tag_manager = self.locator.get_system(TagManager)
            else:
                tag_manager = None
            
            # Get file record
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            # Create hierarchical auto-tags
            auto_tags = []
            
            # General tags: auto/wd_tag/{tag}
            for tag_name in result.get("tags", [])[:30]:  # Limit to top 30
                clean_name = tag_name.replace("_", " ")
                full_name = f"auto/wd_tag/{clean_name}"
                auto_tags.append(full_name)
            
            # Character tags: auto/wd_character/{character}
            for char_name in result.get("characters", []):
                clean_name = char_name.replace("_", " ")
                full_name = f"auto/wd_character/{clean_name}"
                auto_tags.append(full_name)
            
            # Rating: auto/wd_rating/{rating}
            if result.get("rating"):
                rating_name = result["rating"].replace("_", " ")
                auto_tags.append(f"auto/wd_rating/{rating_name}")
            
            # Store tags via TagManager or directly on FileRecord
            if tag_manager:
                for tag_full_name in auto_tags:
                    await tag_manager.add_tag_to_file(file_id, tag_full_name)
            else:
                # Fallback: store in file's auto_tags field
                current_tags = file.auto_tags or []
                updated_tags = list(set(current_tags + auto_tags))
                file.auto_tags = updated_tags
                await file.save()
            
            logger.debug(f"WD-Tagger: Added {len(auto_tags)} tags to {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"WD-Tagger store failed for {file_id}: {e}")
            return False
    
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
