# Event System

USCore uses two distinct event patterns for different use cases.

## 1. EventBus (Async)

The `EventBus` (`src.core.events.bus`) is the primary mechanism for decoupled, application-wide communication.

-   **Type**: Publish/Subscribe
-   **Sync/Async**: Supports both, but primarily designed for `async/await`.
-   **Usage**: Low-frequency, high-value events (e.g., `file.created`, `task.completed`, `scan.started`).

### Publishing

```python
# Async context
await event_bus.publish("file.created", {"path": "/foo.txt"})

# Sync context (e.g., Qt Signal handler)
event_bus.publish_sync("ui.button_clicked", {"id": "btn1"})
```

### Subscribing

```python
def on_file_created(data):
    print(f"File created: {data['path']}")

event_bus.subscribe("file.created", on_file_created)
```

## 2. Signal / ObserverEvent (Sync)

The `Signal` class (`src.core.events.observer`), aliased as `ObserverEvent`, is a synchronous implementation of the Observer pattern, mimicking Qt's Signal/Slot mechanism.

-   **Type**: Multicast Delegate
-   **Sync/Async**: Synchronous only.
-   **Usage**: High-frequency UI updates, property bindings, or where strict synchronous execution is required (e.g., internal component signaling).

### Usage

```python
from src.core.events import Signal

class MyComponent:
    def __init__(self):
        self.on_change = Signal("on_change")

    def do_work(self):
        self.on_change.emit(123)
```
