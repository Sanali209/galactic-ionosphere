# Event System (`src.core.events`)

The event system provides a lightweight implementation of the Observer pattern, designed to decouple components (e.g., Core from UI).

## `ObserverEvent`

A safe publisher-subscriber mechanism.

### Usage

```python
from src.core.events import ObserverEvent

# Define an event
on_process_complete = ObserverEvent("ProcessComplete")

# Subscribe
def my_handler(result):
    print(f"Done: {result}")

on_process_complete.connect(my_handler)

# Emit
on_process_complete.emit("Success")
```

### Features

-   **Error Safety**: If one subscriber raises an exception, others are still executed. Errors are logged.
-   **No Dependencies**: Pure Python implementation.
