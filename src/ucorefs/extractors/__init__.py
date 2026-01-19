"""
UCoreFS Extractors Package.

Provides file processing extractors for Phase 2/3 pipeline.

Phase 2 (batch 20): ThumbnailExtractor, MetadataExtractor, CLIPExtractor, DINOExtractor, WDTaggerExtractor
Phase 3 (batch 1): BLIPExtractor, GroundingDINOExtractor
"""
from src.ucorefs.extractors.base import Extractor
from src.ucorefs.extractors.ai_extractor import AIExtractor
from src.ucorefs.extractors.registry import ExtractorRegistry
from src.ucorefs.extractors.xmp import xmp_extractor, XMPExtractor
from src.ucorefs.extractors.thumbnail import ThumbnailExtractor
from src.ucorefs.extractors.metadata import MetadataExtractor
from src.ucorefs.extractors.clip_extractor import CLIPExtractor
from src.ucorefs.extractors.blip_extractor import BLIPExtractor
from src.ucorefs.extractors.grounding_dino_extractor import GroundingDINOExtractor
from src.ucorefs.extractors.yolo_extractor import YOLOExtractor
from src.ucorefs.extractors.dino_extractor import DINOExtractor
from src.ucorefs.extractors.wd_tagger import WDTaggerExtractor

# Auto-register Phase 2 extractors (batch processing)
ExtractorRegistry.register(ThumbnailExtractor)
ExtractorRegistry.register(MetadataExtractor)
ExtractorRegistry.register(CLIPExtractor)
ExtractorRegistry.register(DINOExtractor)
ExtractorRegistry.register(WDTaggerExtractor)

# Auto-register Phase 3 extractors (one-at-a-time)
ExtractorRegistry.register(BLIPExtractor)
ExtractorRegistry.register(GroundingDINOExtractor)
ExtractorRegistry.register(YOLOExtractor)


__all__ = [
    "Extractor",
    "AIExtractor",
    "ExtractorRegistry",
    "xmp_extractor",
    "XMPExtractor",
    "ThumbnailExtractor",
    "MetadataExtractor",
    "CLIPExtractor",
    "DINOExtractor",
    "BLIPExtractor",
    "GroundingDINOExtractor",
    "YOLOExtractor",
    "WDTaggerExtractor",
]


