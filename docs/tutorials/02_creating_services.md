# Tutorial: Creating a Service

In this tutorial, we will create a simple "GreeterService" that logs a welcome message when the application starts.

## 1. Define the Service

Create a new file `src/core/services/greeter.py`:

```python
from loguru import logger
from src.core.base_system import BaseSystem
from src.core.events.bus import EventBus

class GreeterService(BaseSystem):
    # Declare dependencies
    depends_on = [EventBus]

    async def initialize(self):
        """Called at startup."""
        logger.info("GreeterService initializing...")
        
        # Get dependency
        self.bus = self.locator.get_system(EventBus)
        
        # Subscribe to event
        self.bus.subscribe("app.ready", self.on_app_ready)
        
        await super().initialize()
        logger.info("GreeterService ready!")

    async def on_app_ready(self, data):
        logger.info(f"Welcome to USCore! {data}")
```

## 2. Register the Service

In your application entry point (e.g., `main.py`):

```python
from src.core.bootstrap import ApplicationBuilder
from src.core.services.greeter import GreeterService

def main():
    app = (ApplicationBuilder("My App")
           .with_default_systems()
           .add_system(GreeterService)  # Register here
           .build())
```

## 3. Verify

Run dependencies check in `ServiceLocator`:
-   `EventBus` starts first.
-   `GreeterService` starts second (depends on EventBus).
