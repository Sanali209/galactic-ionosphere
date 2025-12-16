# Configuration System

The Foundation Template uses a reactive configuration manager backed by Pydantic models.

## Access

Config is available globally via the Service Locator or injected into `BaseSystem`s.

```python
# From Locator
current_theme = sl.config.data.general.theme

# From System
self.config.data.mongo.host
```

## Reactivity

The configuration system uses observers to notify components of changes.

```python
# Update a value
# This saves to config.json AND triggers listeners
sl.config.update("general", "theme", "dark")
```

## Schema

Configuration schema is defined in `src/core/config.py`.

```python
class MongoConfig(BaseModel):
    host: str = "mongodb://localhost:27017"
    db_name: str = "app_db"

class GeneralConfig(BaseModel):
    app_name: str = "Foundation App"
    theme: str = "dark"
    log_level: str = "DEBUG"
```
