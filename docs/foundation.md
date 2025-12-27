# Foundation Framework

The Foundation is the bedrock of USCore. It provides the essential plumbing for building scalable desktop applications.

## Service Locator

The `ServiceLocator` is akin to a Dependency Injection container.

### Registering a Service

```python
from src.core.locator import sl
from src.core.base_service import BaseService

class MyService(BaseService):
    def retrieve_data(self):
        return "Data"

# In your bootstrap logic:
sl.register_service(MyService)
```

### Retrieving a Service

```python
my_service = sl.get_service(MyService)
data = my_service.retrieve_data()
```

## Event System

USCore currently supports two event patterns:

1.  **ObserverEvent (Synchronous)**: Best for direct UI updates or tight loops where immediate feedback is required.
    ```python
    from src.core.events.observer import ObserverEvent

    on_data_changed = ObserverEvent("data_changed")
    
    def callback(data):
        print(f"Data changed: {data}")

    on_data_changed.subscribe(callback)
    on_data_changed.emit("New Value")
    ```

2.  **EventBus (Asynchronous)**: Best for decoupled system-wide notifications.
    ```python
    from src.core.events.bus import EventBus
    # Implementation details vary, see `src.core.events.bus`
    ```

## Configuration

Configuration is handled via `config.json`. The `Config` service loads this file at startup.

```json
{
    "app_name": "My App",
    "mongo": {
        "host": "localhost",
        "port": 27017
    }
}
```

Accessing config:
```python
config = sl.get_service(Config)
db_host = config.get("mongo.host", "localhost")
```
