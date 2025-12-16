# Systems Architecture

The Foundation Template is built around a modular **Service Locator** pattern.

## Architecture Diagram (UML)

*(See [system_class.puml](system_class.puml))*

![System Class Diagram](system_class.puml)

## Service Locator (`src.core.locator`)
The `ServiceLocator` is a singleton that manages the lifecycle of all core systems.

- **Initialization**: `sl.init()` sets up the global configuration and event bus.
- **Registration**: `sl.register_system(MySystem)` instantiates the system and injects dependencies.
- **Startup**: `await sl.start_all()` calls `initialize()` on all registered systems sequentially.
- **Shutdown**: `await sl.stop_all()` calls `shutdown()` on all systems in reverse order.

## BaseSystem (`src.core.base_system`)
All major components (AssetManager, Journal, Tasks, etc.) inherit from `BaseSystem`.

```python
class MySystem(BaseSystem):
    async def initialize(self):
        # Perform async setup (DB connections, cache loading)
        await super().initialize()
        
    async def shutdown(self):
        # Cleanup
        await super().shutdown()
```

### Dependency Injection
Every system receives two arguments in `__init__`:
1. `locator`: Access to other systems (e.g. `locator.get_system(JournalService)`).
2. `config`: Access to reactive configuration.

## Lifecycle Sequence

*(See [system_lifecycle.puml](system_lifecycle.puml))*

![System Lifecycle](system_lifecycle.puml)
