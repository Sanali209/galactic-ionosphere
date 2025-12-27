from typing import Any, Dict, Optional, List
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
    task_workers: int = 3  # Number of background task workers

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

class DetectionSettings(BaseModel):
    enabled: bool = False
    backend: str = "yolo"
    yolo: YOLOSettings = Field(default_factory=YOLOSettings)
    mtcnn: MTCNNSettings = Field(default_factory=MTCNNSettings)

class EmbeddingModelSettings(BaseModel):
    enabled: bool = True
    model: Optional[str] = None

class EmbeddingsSettings(BaseModel):
    clip: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)
    dino: EmbeddingModelSettings = Field(default_factory=lambda: EmbeddingModelSettings(model="vit_small_patch16_224.dino"))
    blip: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)

class ProcessingSettings(BaseModel):
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)

class AppConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    ai: AISettings = Field(default_factory=AISettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    library_path: str = "./data"
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)

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
