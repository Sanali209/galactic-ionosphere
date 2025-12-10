from typing import Any
import json
import os
from pydantic import BaseModel, Field
from .events import ObserverEvent

# --- Models ---
class AISettings(BaseModel):
    provider_id: str = "clip_local"
    device: str = "cpu"
    result_limit: int = 20

class MongoSettings(BaseModel):
    host: str = 'localhost'
    port: int = 27017
    database_name: str = "gallery_db"

class AppConfig(BaseModel):
    ai: AISettings = Field(default_factory=AISettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    library_path: str = "./data"

# --- Manager ---
class ConfigManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._data = AppConfig()
        self.on_changed = ObserverEvent("ConfigChanged")
        # Load persisted settings if file exists
        self._load()

    @property
    def data(self) -> AppConfig:
        return self._data

    def update(self, section: str, key: str, value: Any):
        """Update a setting, validate via Pydantic, autosave, and emit change event."""
        section_obj = getattr(self._data, section)
        setattr(section_obj, key, value)
        self._save()
        self.on_changed.emit(section, key, value)

    def _load(self):
        """Load settings from JSON file if present; otherwise keep defaults."""
        if os.path.isfile(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Populate AppConfig from dict (pydantic will validate)
                self._data = AppConfig.parse_obj(raw)
            except Exception as e:
                # If loading fails, keep defaults and log via print (or logger later)
                print(f"Failed to load config from {self.filepath}: {e}")
        else:
            # Ensure file exists with defaults for future saves
            self._save()

    def _save(self):
        """Persist current config to JSON file."""
        try:
            dirname = os.path.dirname(self.filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data.dict(), f, indent=4)
        except Exception as e:
            print(f"Failed to save config to {self.filepath}: {e}")
