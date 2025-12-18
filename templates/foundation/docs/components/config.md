# Configuration System

The Foundation Template uses a reactive configuration manager backed by Pydantic models.

## Access

Config is available globally via the Service Locator or injected into `BaseSystem`s.

```python
# Import
from foundation import sl

# From Locator
current_theme = sl.config.data.general.theme
db_host = sl.config.data.mongo.host

# From BaseSystem
class MySystem(BaseSystem):
    async def initialize(self):
        db_name = self.config.data.mongo.database_name
```

## Reactivity

The configuration system uses observers to notify components of changes.

```python
# Update a value
# This saves to config.json AND triggers listeners
sl.config.update("general", "theme", "dark")

# Subscribe to changes
def on_config_change(section, key, value):
    print(f"{section}.{key} changed to {value}")

sl.config.on_changed.subscribe(on_config_change)
```

## Schema

Configuration schema is defined in `src/core/config.py` using Pydantic models.

```python
from pydantic import BaseModel, Field

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
```

## Example config.json

```json
{
  "general": {
    "debug_mode": true,
    "theme": "dark"
  },
  "ai": {
    "provider_id": "clip_local",
    "device": "cpu",
    "result_limit": 50
  },
  "mongo": {
    "host": "localhost",
    "port": 27017,
    "database_name": "foundation_app"
  },
  "library_path": "./data"
}
```

## Adding Custom Settings

1. **Define Pydantic Model** in `src/core/config.py`:
```python
class MySettings(BaseModel):
    option1: str = "default"
    option2: int = 42
```

2. **Add to AppConfig**:
```python
class AppConfig(BaseModel):
    # ... existing ...
    my_section: MySettings = Field(default_factory=MySettings)
```

3. **Access in code**:
```python
value = sl.config.data.my_section.option1
```

## Persistence

- Configuration is automatically loaded from `config.json` on startup
- Updates via `config.update()` are immediately persisted
- File is created with defaults if it doesn't exist

