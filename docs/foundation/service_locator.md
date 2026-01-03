# Service Locator & System Lifecycle

The **Foundation** framework uses a `ServiceLocator` pattern coupled with a `BaseSystem` interface to manage the application lifecycle and dependencies.

## Service Locator

The `ServiceLocator` (`src.core.locator`) is the central registry for all application services. Unlike a simple global dictionary, it manages:
-   **Dependency Injection**: Ensures systems can access each other.
-   **Startup Order**: Uses topological sort to initialize systems in dependency order.
-   **Safe Shutdown**: Shuts down systems in reverse dependency order.

### Usage

```python
from src.core.locator import sl

# Get a system
db = sl.get_system(DatabaseManager)
```

## Base System

All core services must inherit from `BaseSystem` (`src.core.base_system`).

```python
from src.core.base_system import BaseSystem

class MySystem(BaseSystem):
    # Declare dependencies for topological sort
    depends_on = [DatabaseManager, EventBus]

    async def initialize(self):
        """Called during application startup."""
        logger.info("Starting MySystem")
        self.db = self.locator.get_system(DatabaseManager)
        await super().initialize()

    async def shutdown(self):
        """Called during application shutdown."""
        logger.info("Stopping MySystem")
        await super().shutdown()
```

### Auto-Subscription

`BaseSystem` automatically subscribes marked methods to the `EventBus` if you use the `@subscribe_event` decorator (NOTE: syntactic sugar currently in research/implementation phase, standard usage is manual subscription in `initialize`).

## LifeCycle

1.  **Registration**: Systems are registered via `ApplicationBuilder` or `sl.register_system()`.
2.  **Resolution**: `sl.start_all()` calculates the dependency graph.
3.  **Initialization**: `initialize()` is called on each system in order.
4.  **Running**: The application runs its main loop.
5.  **Shutdown**: `stop_all()` calls `shutdown()` in reverse order.
