# Reactive Configuration (`src.core.config`)

Configuration is managed via Pydantic models for validation and an event system for reactivity.

## Components

### `AppConfig`
A Pydantic model defining the structure of the settings.

```python
class AppConfig(BaseModel):
    ai: AISettings
    mongo: MongoSettings
    library_path: str = "./data"

class AISettings(BaseModel):
    provider_id: str
    device: str
    result_limit: int
```

### `ConfigManager`
Manages access to `AppConfig` and emits events when settings change.

#### Updating Settings

**Do not modify `sl.config.data` directly.** Use the `update()` method to ensure events are fired.

```python
# Correct
sl.config.update("ai", "provider_id", "gpt4")

# This triggers:
# This triggers:
# 1. Pydantic Validation
# 2. Auto-save to config.json
# 3. Event: sl.config.on_changed(section, key, value)
```
