from typing import Any
from pydantic import BaseModel, Field
from .events import ObserverEvent

# --- Models ---
class AISettings(BaseModel):
    provider_id: str = "clip_local"
    device: str = "cpu"

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
        self._data = AppConfig() # + load logic (TODO: implement json load)
        
        # Event: (section, key, new_value)
        self.on_changed = ObserverEvent("ConfigChanged")

    @property
    def data(self) -> AppConfig:
        return self._data

    def update(self, section: str, key: str, value: Any):
        """
        The only way to modify settings.
        Guarantees notification of all systems.
        """
        # 1. Reflection to access nested fields
        section_obj = getattr(self._data, section)
        
        # 2. Pydantic validation (will raise error if type is incorrect)
        setattr(section_obj, key, value)
        
        # 3. Save (autosave)
        self._save()
        
        # 4. Notify
        self.on_changed.emit(section, key, value)

    def _save(self):
        # json dump logic
        pass
