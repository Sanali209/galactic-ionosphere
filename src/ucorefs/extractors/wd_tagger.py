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

from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.models.file_record import FileRecord


class WDTaggerExtractor(AIExtractor):
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
    batch_supported: bool = True  # Supports batching via parallelism
    is_cpu_heavy: bool = True  # SAN-14: AI inference with image preprocessing (PIL operations)
    needs_model: bool = True  # Requires WD-Tagger model
    
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
    
    # Inherited from AIExtractor: _get_llm_service(), extract()
    
    async def _extract_via_worker(self, files: List[FileRecord], llm_service) -> Dict[ObjectId, Any]:
        """Extract tags using LLMWorkerService."""
        from src.core.llm.models import LLMJobRequest
        
        file_paths = [str(f.path) for f in files]
        path_to_id = {str(f.path): f._id for f in files}
        
        request = LLMJobRequest(
            task_type="wdtagger",
            file_paths=file_paths,
            options={
                "general_threshold": self.general_threshold,
                "character_threshold": self.character_threshold,
                "model_repo": self.model_repo
            }
        )
        
        try:
            future = await llm_service.submit_job(request)
            result = await future
            
            if not result.success:
                logger.warning(f"WDTagger worker job failed: {result.error}, falling back to legacy")
                return await self._extract_legacy(files)
            
            # Check if worker returned actual data (not just None placeholders)
            results = {}
            has_real_data = False
            for file_path, data in (result.data or {}).items():
                file_id = path_to_id.get(file_path)
                if file_id:
                    results[file_id] = data
                    if data is not None:
                        has_real_data = True
            
            # If worker returned all None (stub not implemented), fall back to legacy
            if not has_real_data and results:
                logger.debug("[WDTagger] Worker returned empty data, falling back to legacy")
                return await self._extract_legacy(files)
            
            logger.info(f"[WDTagger] Processed {len(results)}/{len(files)} via LLMWorkerService")
            return results
            
        except Exception as e:
            logger.error(f"WDTagger extraction via service failed: {e}")
            return await self._extract_legacy(files)
    
    async def _extract_legacy(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """Legacy: Extract tags using WDTaggerService with semaphore concurrency."""
        import asyncio
        results = {}
        
        service = self._get_tagger_service()
        if not service:
            logger.warning("WDTaggerService not available, skipping tagging")
            return {f._id: None for f in files}
        
        # Check if service is ready
        if not service.is_ready:
            logger.warning(f"WDTaggerService not ready (is_ready={service.is_ready}), skipping tagging")
            return {f._id: None for f in files}
        
        ai_workers = 4
        if self.locator and self.locator.config:
            try:
                ai_workers = self.locator.config.data.processing.ai_workers
            except Exception:
                pass
        
        semaphore = asyncio.Semaphore(ai_workers)
        logger.debug(f"WDTagger: Processing batch of {len(files)} with concurrency {ai_workers}")

        async def _process_single(file_record):
            async with semaphore:
                try:
                    tags_result = await service.tag_image(Path(file_record.path))
                    if tags_result:
                        tag_count = len(tags_result.get("tags", []))
                        logger.debug(f"WDTagger: {file_record.name} -> {tag_count} tags")
                    return file_record._id, tags_result
                except Exception as e:
                    logger.error(f"WD-Tagger failed for {file_record.path}: {e}")
                    return file_record._id, None

        tasks = [_process_single(f) for f in files]
        batch_results = await asyncio.gather(*tasks)
        
        success = sum(1 for _, r in batch_results if r is not None)
        logger.info(f"WDTagger: {success}/{len(files)} files tagged successfully")
        
        for fid, res in batch_results:
            results[fid] = res
        
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
        import asyncio
        
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
                for i, tag_full_name in enumerate(auto_tags):
                    await tag_manager.add_tag_to_file(file_id, tag_full_name)
                    # Yield to UI every 5 tags to prevent blocking
                    if (i + 1) % 5 == 0:
                        await asyncio.sleep(0)
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
