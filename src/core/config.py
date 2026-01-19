from typing import Any, Dict, List, Optional
import json
import os
from pydantic import BaseModel, Field
from loguru import logger
from .events import Signal

# --- Generic Settings Models ---
class AISettings(BaseModel):
    provider_id: str = "clip_local"
    device: str = "cpu"
    result_limit: int = 50

class MongoSettings(BaseModel):
    host: str = 'localhost'
    port: int = 27017
    database_name: str = "foundation_app"

class GeneralSettings(BaseModel):
    debug_mode: bool = True
    theme: str = "dark"
    task_workers: int = 8  # Number of background task workers (orchestrators)

# --- Detection/Processing Settings ---
class YOLOSettings(BaseModel):
    enabled: bool = False
    model: str = "yolov8n.pt"
    confidence: float = 0.25
    classes: Optional[List[str]] = None
    use_gpu: bool = True

class MTCNNSettings(BaseModel):
    enabled: bool = False
    min_face_size: int = 20
    thresholds: List[float] = Field(default_factory=lambda: [0.6, 0.7, 0.7])
    use_gpu: bool = True

class GroundingDINOSettings(BaseModel):
    """GroundingDINO zero-shot object detection settings."""
    enabled: bool = True
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    class_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "person": "person", "face": "face", "cat": "cat", "dog": "dog",
        "car": "vehicle", "truck": "vehicle", "motorcycle": "vehicle",
        "building": "architecture", "text": "text", "logo": "logo"
    })

class DetectionSettings(BaseModel):
    enabled: bool = False
    backend: str = "yolo"
    yolo: YOLOSettings = Field(default_factory=YOLOSettings)
    mtcnn: MTCNNSettings = Field(default_factory=MTCNNSettings)
    grounding_dino: GroundingDINOSettings = Field(default_factory=GroundingDINOSettings)

class WDTaggerSettings(BaseModel):
    """WD-Tagger auto-tagging settings."""
    enabled: bool = True
    model_repo: str = "SmilingWolf/wd-vit-tagger-v3"
    general_threshold: float = 0.35
    character_threshold: float = 0.6

class EmbeddingModelSettings(BaseModel):
    enabled: bool = True
    model: Optional[str] = None

class EmbeddingsSettings(BaseModel):
    clip: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)
    dino: EmbeddingModelSettings = Field(default_factory=lambda: EmbeddingModelSettings(model="vit_small_patch16_224.dino"))
    blip: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)

class UISettings(BaseModel):
    """UI layout and state persistence settings."""
    window_state: Optional[str] = None
    window_width: int = 1400
    window_height: int = 900
    recent_directories: List[str] = Field(default_factory=list)

class ThumbnailSettings(BaseModel):
    """Thumbnail generation and caching settings."""
    default_size: int = 128
    sizes: List[int] = Field(default_factory=lambda: [128, 256, 512])
    cache_path: Optional[str] = None
    max_cache_mb: int = 1000
    jpeg_quality: int = 85

class SearchSettings(BaseModel):
    """Search and FAISS settings."""
    default_limit: int = 100
    min_similarity: float = 0.7
    faiss_index_type: str = "Flat"

class SimilaritySettings(BaseModel):
    """Similarity search threshold settings."""
    threshold: float = 0.7

class LLMSettings(BaseModel):
    """LLM provider settings."""
    enabled: bool = False
    provider: str = "openai"
    model: Optional[str] = None

class LLMWorkerModelSettings(BaseModel):
    """Per-model enablement for LLM workers."""
    enabled: bool = True

class LLMWorkerSettings(BaseModel):
    """LLM worker pool settings for non-blocking inference."""
    enabled: bool = False  # Enable/disable LLM worker process
    num_workers: int = 2  # Number of worker processes
    queue_max_size: int = 100
    idle_unload_seconds: int = 300  # Unload models after idle time
    models: Dict[str, LLMWorkerModelSettings] = Field(default_factory=lambda: {
        "clip": LLMWorkerModelSettings(enabled=True),
        "blip": LLMWorkerModelSettings(enabled=True),
        "wdtagger": LLMWorkerModelSettings(enabled=True),
        "yolo": LLMWorkerModelSettings(enabled=False),
        "grounding_dino": LLMWorkerModelSettings(enabled=False),
    })

class ProcessingSettings(BaseModel):
    process_workers: int = 4  # Number of process pool workers for CPU-heavy non-LLM tasks
    ai_workers: int = 4  # Number of concurrent AI threads (CPU heavy)
    batch_chunk_size: int = 5  # Files to process before yielding to UI
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    wd_tagger: WDTaggerSettings = Field(default_factory=WDTaggerSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)

class MetadataSettings(BaseModel):
    auto_fill_description_from_blip: bool = True
    prefer_xmp_over_existing: bool = False

class AppConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    ai: AISettings = Field(default_factory=AISettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    ui: UISettings = Field(default_factory=UISettings)
    library_path: str = "./data"
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    thumbnail: ThumbnailSettings = Field(default_factory=ThumbnailSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    similarity: SimilaritySettings = Field(default_factory=SimilaritySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    llm_workers: LLMWorkerSettings = Field(default_factory=LLMWorkerSettings)

# --- Manager ---
class ConfigManager:
    """
    Manages application configuration with persistence and reactivity.
    """
    def __init__(self, filepath: str = "config.json"):
        self.filepath = filepath
        self._data = AppConfig()
        self.on_changed = Signal("ConfigChanged")
        self._load()

    @property
    def data(self) -> AppConfig:
        return self._data

    def update(self, section: str, key: str, value: Any):
        """Update a setting, validate via Pydantic, autosave, and emit change event."""
        if not hasattr(self._data, section):
            raise ValueError(f"Invalid section: {section}")
        
        section_obj = getattr(self._data, section)
        if not hasattr(section_obj, key):
             raise ValueError(f"Invalid key: {key} in section {section}")
             
        setattr(section_obj, key, value)
        self._save()
        self.on_changed.emit(section, key, value)

    def get(self, section: str, key: str) -> Any:
        section_obj = getattr(self._data, section)
        return getattr(section_obj, key)

    def _load(self):
        """Load settings from JSON or TOML file if present; otherwise keep defaults."""
        if os.path.isfile(self.filepath):
            try:
                if self.filepath.endswith('.toml'):
                    import tomllib
                    with open(self.filepath, "rb") as f:
                        raw = tomllib.load(f)
                else:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                self._data = AppConfig.model_validate(raw)
            except Exception as e:
                logger.error(f"Failed to load config from {self.filepath}: {e}")
                self._save()
        else:
            self._save()

    def _save(self):
        """Persist current config to JSON file."""
        try:
            dirname = os.path.dirname(self.filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data.model_dump(), f, indent=4) # Pydantic v2
        except Exception as e:
            logger.error(f"Failed to save config to {self.filepath}: {e}")

    # --- BaseSystem Interface ---
    depends_on = [] 

    async def initialize(self):
        """No-op init for compatibility with ServiceLocator."""
        pass
        
    async def shutdown(self):
        """No-op shutdown."""
        pass
