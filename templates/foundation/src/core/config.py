from typing import Any, Dict
import json
import os
from pydantic import BaseModel, Field
from .events import ObserverEvent

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

class AppConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    ai: AISettings = Field(default_factory=AISettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    library_path: str = "./data"

# --- Manager ---
class ConfigManager:
    """
    Manages application configuration with persistence and reactivity.
    """
    def __init__(self, filepath: str = "config.json"):
        self.filepath = filepath
        self._data = AppConfig()
        self.on_changed = ObserverEvent("ConfigChanged")
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
        """Load settings from JSON file if present; otherwise keep defaults."""
        if os.path.isfile(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = AppConfig.model_validate(raw) # Pydantic v2 style, or parse_obj for v1
            except Exception as e:
                print(f"Failed to load config from {self.filepath}: {e}")
                self._save() # overwrites with defaults if corrupted? or maybe better to backup.
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
            print(f"Failed to save config to {self.filepath}: {e}")
