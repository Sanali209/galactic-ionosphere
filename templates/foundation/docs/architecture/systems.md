# Systems Architecture

The Foundation Template is built around a modular **Service Locator** pattern with simplified bootstrap helpers.

## Quick Start: ApplicationBuilder (v0.1.0+)

The new `ApplicationBuilder` provides a fluent API for setting up applications:

```python
from foundation import ApplicationBuilder, run_app

builder = (ApplicationBuilder("My App", "config.json")
           .with_default_systems()        # Register foundation systems
           .with_logging(True)             # Setup logging
           .add_system(MyCustomService))   # Add custom systems

run_app(MainWindow, MainViewModel, builder=builder)
```

## Service Locator (`src.core.locator`)

The `ServiceLocator` is a singleton that manages the lifecycle of all core systems.

### Manual Usage (Advanced)

```python
from foundation import sl, DatabaseManager, TaskSystem

# Initialize
sl.init("config.json")

# Register systems
sl.register_system(DatabaseManager)
sl.register_system(TaskSystem)

# Start all
await sl.start_all()

# Access systems
db = sl.get_system(DatabaseManager)

# Shutdown
await sl.stop_all()
```

### Methods

- **`init(config_path)`**: Sets up configuration and event bus
- **`register_system(SystemClass)`**: Instantiates and injects dependencies
- **`get_system(SystemClass)`**: Retrieve registered system instance
- **`start_all()`**: Calls `initialize()` on all systems sequentially
- **`stop_all()`**: Calls `shutdown()` in reverse order

## BaseSystem (`src.core.base_system`)

All major components inherit from `BaseSystem`:

```python
from foundation import BaseSystem

class MyService(BaseSystem):
    async def initialize(self):
        # Perform async setup (DB connections, cache loading)
        logger.info(f"{self.__class__.__name__} initializing")
        
        # Access other systems
        db = self.locator.get_system(DatabaseManager)
        
        # Access config
        setting = self.config.data.my_section.some_value
        
        await super().initialize()  # Sets _is_ready = True
        
    async def shutdown(self):
        # Cleanup
        logger.info(f"{self.__class__.__name__} shutting down")
        await super().shutdown()
```

### Dependency Injection

Every `BaseSystem` receives in `__init__`:

1. **`locator`**: ServiceLocator instance to access other systems
2. **`config`**: ConfigManager instance for reactive configuration

### Properties

- **`is_ready`**: Boolean indicating system initialization status

## Default Foundation Systems

When using `ApplicationBuilder.with_default_systems()`, these are registered:

1. **DatabaseManager** - MongoDB connection and collection access
2. **CommandBus** - Command/handler pattern for business logic
3. **JournalService** - Structured logging to database
4. **AssetManager** - File and asset handling
5. **TaskSystem** - Background task queue with persistence

## Lifecycle Sequence

```
Application Start
    ├─> ApplicationBuilder configuration
    ├─> sl.init(config_path)
    ├─> System Registration (order matters)
    ├─> await sl.start_all()
    │   ├─> DatabaseManager.initialize()
    │   ├─> CommandBus.initialize()
    │   ├─> JournalService.initialize()
    │   ├─> AssetManager.initialize()
    │   ├─> TaskSystem.initialize()
    │   └─> CustomSystem.initialize()
    ├─> UI Creation (MainWindow, ViewModels)
    └─> Qt Event Loop

Application Shutdown
    ├─> await sl.stop_all()
    │   ├─> CustomSystem.shutdown()
    │   ├─> TaskSystem.shutdown()
    │   ├─> AssetManager.shutdown()
    │   ├─> JournalService.shutdown()
    │   ├─> CommandBus.shutdown()
    │   └─> DatabaseManager.shutdown()
    └─> Exit
```

## Creating Custom Systems

```python
from foundation import BaseSystem, sl
from loguru import logger

class NotificationService(BaseSystem):
    """Sends notifications via various channels."""
    
    async def initialize(self):
        """Setup notification channels."""
        logger.info("Initializing NotificationService")
        
        # Access other systems if needed
        self.journal = self.locator.get_system(JournalService)
        
        # Your initialization logic
        self.channels = []
        
        await super().initialize()
    
    async def send(self, message: str):
        """Send a notification."""
        if not self.is_ready:
            logger.warning("Service not ready")
            return
            
        # Send logic here
        await self.journal.log("notification_sent", {"message": message})
    
    async def shutdown(self):
        """Cleanup channels."""
        logger.info("Shutting down NotificationService")
        self.channels.clear()
        await super().shutdown()

# Use it
builder = (ApplicationBuilder("My App")
           .with_default_systems()
           .add_system(NotificationService))
```

## Best Practices

1. **Register dependent systems first** (e.g., Database before systems that use it)
2. **Use `ApplicationBuilder`** instead of manual setup
3. **Always call `super().initialize()`** and `super().shutdown()`
4. **Check `is_ready`** before using system methods
5. **Use `locator.get_system()`** for cross-system communication

