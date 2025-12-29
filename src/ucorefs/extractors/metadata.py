"""
UCoreFS - Metadata Extractor

Extracts file metadata including EXIF, XMP, and resolves tags.
Phase 2 extractor - runs in batches of 20.
"""
from datetime import datetime
from typing import List, Dict, Any
from bson import ObjectId
from loguru import logger

from src.ucorefs.extractors.base import Extractor
from src.ucorefs.extractors.xmp import xmp_extractor
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState


class MetadataExtractor(Extractor):
    """
    Extracts and applies file metadata.
    
    Handles:
    - XMP metadata (rating, label, description, tags)
    - Tag resolution with synonyms via RulesEngine
    - EXIF data for images
    """
    
    name = "metadata"
    phase = 2
    priority = 90  # After thumbnails
    batch_supported = True
    
    # Supported file types - expanded to all pyexiv2-compatible formats
    # Images: JPG, PNG, TIFF, WebP, GIF, BMP
    # RAW: CR2, NEF, ARW, DNG, RAF, ORF, etc.
    # Others: PSD, PDF, EPS
    SUPPORTED_EXTENSIONS = {
        # Standard images
        'jpg', 'jpeg', 'png', 'tiff', 'tif', 'webp', 'gif', 'bmp',
        # RAW formats
        'cr2', 'cr3', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'pef', 'srw',
        # Professional formats
        'psd', 'pdf', 'eps'
    }
    
    async def extract(self, files: List[FileRecord]) -> Dict[ObjectId, Any]:
        """
        Extract metadata from files.
        
        Args:
            files: List of files to process
            
        Returns:
            Dict mapping file_id -> metadata dict
        """
        results = {}
        
        for file in files:
            if not self.can_process(file):
                continue
            
            metadata = {}
            
            try:
                # Extract XMP metadata (works with all supported file types)
                if xmp_extractor.is_available():
                    xmp_data = xmp_extractor.extract(file.path)
                    
                    if xmp_data:
                        metadata["label"] = xmp_data.get("label", "")
                        metadata["description"] = xmp_data.get("description", "")
                        metadata["raw_tags"] = xmp_data.get("tags", [])
                        
                        # Extract rating from XMP if present
                        raw_xmp = xmp_data.get("raw_xmp", {})
                        if "Xmp.xmp.Rating" in raw_xmp:
                            try:
                                metadata["rating"] = int(raw_xmp["Xmp.xmp.Rating"])
                            except (ValueError, TypeError):
                                pass
                
                if metadata:
                    results[file._id] = metadata
                    
            except Exception as e:
                logger.error(f"Metadata extraction failed for {file._id}: {e}")
        
        return results
    
    async def store(self, file_id: ObjectId, result: Any) -> bool:
        """
        Store extracted metadata to FileRecord.
        
        Optionally fills empty fields or overwrites based on config.
        Also resolves raw tags to TagManager with synonym support.
        """
        try:
            file = await FileRecord.get(file_id)
            if not file:
                return False
            
            # Get metadata config from global system
            prefer_xmp = False
            if self.locator and hasattr(self.locator, 'config'):
                try:
                    meta_config = self.locator.config.data.metadata
                    prefer_xmp = getattr(meta_config, 'prefer_xmp_over_existing', False)
                except Exception:
                    pass

            # Update Metadata fields
            if "label" in result and result["label"]:
                if prefer_xmp or not file.label:
                    file.label = result["label"]
                    logger.debug(f"Set label='{file.label}' from XMP for {file.name}")
            
            if "description" in result and result["description"]:
                if prefer_xmp or not file.description:
                    file.description = result["description"]
                    logger.debug(f"Set description from XMP for {file.name}")
            
            if "rating" in result and result["rating"]:
                if prefer_xmp or file.rating == 0:
                    file.rating = result["rating"]
                    logger.debug(f"Set rating={file.rating} from XMP for {file.name}")
            
            # Resolve tags with synonym support
            raw_tags = result.get("raw_tags", [])
            if raw_tags:
                resolved_ids = await self._resolve_tags(raw_tags)
                # Merge with existing tags (don't overwrite)
                existing = set(file.tag_ids)
                existing.update(resolved_ids)
                file.tag_ids = list(existing)
                logger.debug(f"Added {len(resolved_ids)} tags from XMP for {file.name}")
            
            # Update processing state
            if file.processing_state < ProcessingState.METADATA_READY:
                file.processing_state = ProcessingState.METADATA_READY
            
            file.last_processed_at = datetime.now()
            await file.save()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metadata for {file_id}: {e}")
            return False
    
    async def _resolve_tags(self, raw_tags: List[str]) -> List[ObjectId]:
        """
        Resolve raw tag strings to TagManager records.
        
        Uses RulesEngine for synonym resolution if available.
        """
        from src.ucorefs.tags import TagManager
        
        resolved_ids = []
        
        # Try to get RulesEngine for synonym resolution
        rules_engine = None
        if self.locator:
            try:
                from src.ucorefs.rules import RulesEngine
                rules_engine = self.locator.get_system(RulesEngine)
            except (KeyError, ImportError):
                pass
        
        # Get TagManager
        tag_manager = None
        if self.locator:
            try:
                tag_manager = self.locator.get_system(TagManager)
            except (KeyError, ImportError):
                pass
        
        for raw_tag in raw_tags:
            try:
                # Resolve synonym if RulesEngine available
                canonical = raw_tag
                if rules_engine and hasattr(rules_engine, 'resolve_tag_synonym'):
                    try:
                        canonical = await rules_engine.resolve_tag_synonym(raw_tag)
                    except Exception:
                        pass
                
                # Get or create tag
                if tag_manager and hasattr(tag_manager, 'get_or_create_by_name'):
                    tag = await tag_manager.get_or_create_by_name(canonical)
                    if tag:
                        resolved_ids.append(tag._id)
                        
            except Exception as e:
                logger.debug(f"Failed to resolve tag '{raw_tag}': {e}")
        
        return resolved_ids
    
    def can_process(self, file: FileRecord) -> bool:
        """Process files with XMP-compatible extensions."""
        if not file.extension:
            return False
        return file.extension.lower().lstrip('.') in self.SUPPORTED_EXTENSIONS
