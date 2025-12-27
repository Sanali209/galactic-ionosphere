# SLM Core Architecture - Detailed Documentation

## Table of Contents
1. [Overview](#overview)
2. [Singleton Pattern](#singleton-pattern)
3. [Lifecycle Management](#lifecycle-management)
4. [Dependency Injection](#dependency-injection)
5. [Message Bus](#message-bus)
6. [Configuration System](#configuration-system)
7. [Component System](#component-system)
8. [Plugin System](#plugin-system)
9. [Decorators](#decorators)

---

## Overview

### Design Philosophy

The SLM Core framework follows these principles:

1. **Pythonic** - Uses Python idioms and patterns (metaclasses, decorators, context managers)
2. **Elegant** - Simple, clean API with minimal boilerplate
3. **Type-safe** - Full type hints for IDE support
4. **Thread-safe** - Proper locking and synchronization
5. **Testable** - Easy mocking and reset capabilities
6. **Event-driven** - Reactive configuration and loose coupling

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Application                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │            LifecycleManager                       │  │
│  │  (State Machine: CREATED → ... → SHUTDOWN)       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│   Config     │ │ MessageBus  │ │Dependencies│
│  (Singleton) │ │ (Singleton) │ │ (Singleton)│
└──────────────┘ └─────────────┘ └────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ Components   │ │   Plugins   │ │  Services  │
│  (Managed)   │ │  (Dynamic)  │ │(Registered)│
└──────────────┘ └─────────────┘ └────────────┘
```

---

## Singleton Pattern

### Implementation

**File:** `singleton.py`

The framework uses a thread-safe metaclass-based singleton pattern:

```python
class SingletonMeta(type):
    """
    Thread-safe singleton metaclass
    """
    _instances: Dict[Type, object] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
```

### Key Features

1. **Thread-safe**: Uses double-check locking pattern
2. **Lazy initialization**: Created on first access
3. **Type-based**: One instance per class type
4. **Resettable**: For testing purposes

### Usage Patterns

#### Basic Singleton

```python
from SLM.core.singleton import Singleton

class MyService(Singleton):
    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
        
        self.data = {}
        self._initialized = True
```

#### Multiple Access Points

```python
# All return the same instance
s1 = MyService()
s2 = MyService()
s3 = MyService.instance()

assert s1 is s2 is s3  # True
```

#### Testing with Reset

```python
# In tests
from SLM.core import reset_all

def test_something():
    config.setting = "test_value"
    
    # Test logic...
    
    # Clean up
    reset_all()  # Resets all singletons
```

### Thread Safety

The singleton pattern ensures thread-safe initialization:

```python
import threading

def create_singleton():
    return Config()

threads = [threading.Thread(target=create_singleton) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Only one instance is created, no race conditions
```

---

## Lifecycle Management

### State Machine

**File:** `lifecycle.py`

The lifecycle follows a strict state machine:

```
CREATED
   ↓
CONFIGURED
   ↓
INITIALIZED ←──┐
   ↓           │
STARTED        │
   ↓           │
RUNNING        │
   ↓           │
STOPPING       │
   ↓           │
STOPPED ───────┘
   ↓
SHUTDOWN
```

### States Explained

| State | Description | Entry Conditions | Exit Actions |
|-------|-------------|------------------|--------------|
| **CREATED** | Initial state after instantiation | Automatic | None |
| **CONFIGURED** | Configuration loaded | config.update() called | Config validated |
| **INITIALIZED** | Dependencies resolved | setup_dependencies() | Components initialized |
| **STARTED** | Components started | start() called | Resources allocated |
| **RUNNING** | Main loop active | run() called | Event processing |
| **STOPPING** | Graceful shutdown initiated | stop() called | Cleanup started |
| **STOPPED** | All stopped | All cleanup done | Resources released |
| **SHUTDOWN** | Final cleanup | shutdown() called | Everything cleaned |

### Lifecycle Manager

```python
from SLM.core.lifecycle import LifecycleManager, AppState

lifecycle = LifecycleManager()

# Check state
if lifecycle.is_initialized:
    print("Ready to start")

# Transition states
lifecycle.transition_to(AppState.STARTED)

# Register hooks
def on_start():
    print("Starting...")

lifecycle.register_hook(AppState.STARTED, on_start)
```

### State Guards

Prevent invalid transitions:

```python
if lifecycle.can_start:
    lifecycle.transition_to(AppState.STARTED)
else:
    print("Cannot start - not initialized")
```

### Lifecycle Hooks

Execute code at specific lifecycle points:

```python
from SLM.core import on_app_start, on_app_stop

@on_app_start
def initialize_database():
    """Called when app enters STARTED state"""
    db.connect()

@on_app_stop
def cleanup_database():
    """Called when app enters STOPPED state"""
    db.disconnect()
```

---

## Dependency Injection

### Container Pattern

**File:** `dependency.py`

The framework uses a service locator pattern with automatic dependency injection:

```python
┌─────────────────────────────────────┐
│      DependencyManager              │
│  ┌───────────────────────────────┐  │
│  │    DependencyContainer        │  │
│  │  - Services                   │  │
│  │  - Factories                  │  │
│  │  - Singletons                 │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │    DependencyInjector         │  │
│  │  - inject_dependencies()      │  │
│  │  - create_instance()          │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │    DependencyGraph            │  │
│  │  - Topological sort           │  │
│  │  - Cycle detection            │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Registration Types

#### 1. Singleton Registration

```python
from SLM.core import dependencies

# Register singleton (same instance always)
db_service = DatabaseService()
dependencies.register_singleton(DatabaseService, db_service)
```

#### 2. Service Registration

```python
# Register service (cached after first creation)
cache_service = CacheService()
dependencies.register_service(CacheService, cache_service)
```

#### 3. Factory Registration

```python
# Register factory (new instance each time)
def create_logger():
    return Logger(level="INFO")

dependencies.register_factory(Logger, create_logger)
```

### Dependency Injection Methods

#### 1. Type Annotation Injection

```python
class MyComponent(Component):
    # Declare dependencies with type hints
    config: Config
    message_bus: MessageBus
    
    def on_start(self):
        # Dependencies automatically injected
        print(f"Host: {self.config.database.host}")
```

#### 2. Decorator Injection

```python
from SLM.core import inject

@inject(config=Config, bus=MessageBus)
def process_data(data: str, config, bus):
    bus.publish("processing", data=data)
    return f"Processed with {config.app.name}"
```

#### 3. Auto-Injection

```python
from SLM.core import auto_inject

@auto_inject
def smart_function(data: str, config: Config, bus: MessageBus):
    # config and bus automatically injected based on type hints
    pass
```

### Dependency Resolution

The framework automatically resolves dependencies in the correct order:

```python
# Define components with dependencies
class ServiceA(Component):
    pass  # No dependencies

class ServiceB(Component):
    service_a: ServiceA  # Depends on A

class ServiceC(Component):
    service_b: ServiceB  # Depends on B

# Framework initializes in order: A → B → C
dependencies.setup_dependencies()
```

### Cycle Detection

Circular dependencies are automatically detected:

```python
class ServiceX(Component):
    service_y: 'ServiceY'  # Depends on Y

class ServiceY(Component):
    service_x: ServiceX  # Depends on X

# Raises: ValueError: Circular dependencies detected
dependencies.setup_dependencies()
```

---

## Message Bus

### Event-Driven Architecture

**File:** `message_bus.py`

The message bus provides loose coupling between components:

```python
┌──────────────────────────────────────┐
│          MessageBus                  │
│  ┌────────────────────────────────┐  │
│  │  Subscribers                   │  │
│  │  {                             │  │
│  │    'event.type': [handler1,    │  │
│  │                   handler2]     │  │
│  │  }                             │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  Message Queue                 │  │
│  │  [message1, message2, ...]     │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### Publishing Events

```python
from SLM.core import bus

# Simple publish
bus.publish("user.created", user_id=123, username="john")

# With structured data
bus.publish("order.placed", 
    order_id=456,
    items=["item1", "item2"],
    total=99.99,
    timestamp=time.time()
)
```

### Subscribing to Events

#### Manual Subscription

```python
def on_user_created(event_type: str, **data):
    user_id = data['user_id']
    print(f"User {user_id} created")

bus.subscribe("user.created", on_user_created)
```

#### Decorator Subscription

```python
from SLM.core import subscribe

@subscribe("user.created", "user.updated")
def handle_user_events(event_type: str, **data):
    if event_type == "user.created":
        print("User created")
    elif event_type == "user.updated":
        print("User updated")
```

### Message Structure

Every message has this structure:

```python
{
    'id': 'unique-uuid',
    'type': 'event.name',
    'timestamp': 1234567890.123,
    'data': {
        'key1': 'value1',
        'key2': 'value2'
    }
}
```

### Background Processing

```python
# Start async processing
bus.start_processing()

# Messages are queued and processed in background

# Stop processing
bus.stop_processing()
```

### Message Patterns

#### Request-Response

```python
response = bus.publish_and_wait(
    message_type="query.user",
    response_type="query.user.response",
    timeout=5.0,
    user_id=123
)
```

#### Waiting for Events

```python
# Wait for specific event
message = bus.wait_for_message("app.ready", timeout=10.0)
if message:
    print("App is ready!")
```

---

## Configuration System

### Dynamic Configuration

**File:** `config.py`

The configuration system provides multiple access patterns:

### Access Patterns

#### 1. Attribute Style

```python
from SLM.core import config

config.database.host = "localhost"
config.database.port = 5432
config.database.user = "admin"

print(config.database.host)  # "localhost"
```

#### 2. Dictionary Style

```python
config["database"] = {
    "host": "localhost",
    "port": 5432
}

print(config["database"]["host"])  # "localhost"
```

#### 3. Dot-Separated Keys

```python
config.set_value("database.host", "localhost")
host = config.get("database.host", default="127.0.0.1")
```

### Nested Configuration

```python
# Deep nesting
config.app.features.caching.enabled = True
config.app.features.caching.ttl = 3600
config.app.features.caching.backend = "redis"

# Access nested values
if config.app.features.caching.enabled:
    ttl = config.app.features.caching.ttl
```

### Configuration Updates

```python
# Update with dictionary
config.update({
    "logging": {
        "level": "INFO",
        "format": "json",
        "handlers": ["console", "file"]
    }
})

# Merge configurations
config.merge(other_config)
```

### Reactive Configuration

Configuration changes automatically trigger events:

```python
from SLM.core import on_config_change

@on_config_change("database.host")
def on_db_host_changed(old_value, new_value):
    print(f"DB host changed: {old_value} → {new_value}")
    reconnect_database(new_value)

# Changing config triggers the handler
config.database.host = "new-host"  # Handler called!
```

### Change Tracking

```python
# Temporarily disable tracking
config.disable_change_tracking()
config.database.host = "test-host"  # No event fired
config.enable_change_tracking()

# Or use context manager style
with config.set_with_tracking_disabled():
    config.database.host = "test-host"
```

### Conversion Methods

```python
# Convert to dictionary
config_dict = config.to_dict()

# Create copy
config_copy = config.copy()

# Clear all values
config.clear()
```

---

## Component System

### Component Lifecycle

**File:** `component.py`

Components have their own lifecycle within the app lifecycle:

```
Component Lifecycle:
  new() → initialize() → start() → [update()] → stop() → shutdown()
```

### Creating Components

```python
from SLM.core import Component, component

@component(name="DatabaseService")
class DatabaseService(Component):
    # Declare dependencies
    config: Config
    message_bus: MessageBus
    
    def on_initialize(self):
        """Called once during initialization"""
        self.connection = None
        print("DB service initialized")
    
    def on_start(self):
        """Called when starting"""
        host = self.config.database.host
        self.connection = connect(host)
        self.message_bus.publish("db.connected")
    
    def on_update(self):
        """Called periodically (if update loop enabled)"""
        self.check_connection()
    
    def on_stop(self):
        """Called when stopping"""
        if self.connection:
            self.connection.close()
        self.message_bus.publish("db.disconnected")
    
    def on_shutdown(self):
        """Called during final cleanup"""
        self.connection = None
```

### Component Manager

```python
from SLM.core import components

# Register component
db_service = DatabaseService()
components.register_component("database", db_service)

# Get component
db = components.get_component("database")

# Lifecycle management
components.initialize_all()
components.start_all()
components.stop_all()
components.shutdown_all()
```

### Component Communication

#### Via Message Bus

```python
class ServiceA(Component):
    def on_start(self):
        self.send_message("service.a.ready", status="ok")

class ServiceB(Component):
    def on_initialize(self):
        self.subscribe_to_message("service.a.ready", self.on_a_ready)
    
    def on_a_ready(self, event_type, **data):
        print(f"Service A is ready: {data['status']}")
```

#### Direct Access

```python
class ServiceC(Component):
    service_a: ServiceA  # Injected
    
    def do_something(self):
        self.service_a.some_method()
```

---

## Plugin System

### Plugin Architecture

**File:** `plugin_system.py`

Plugins extend application functionality dynamically:

### Creating Plugins

```python
from SLM.core import Plugin

class MyPlugin(Plugin):
    def on_load(self):
        """Called when plugin is loaded"""
        print(f"Plugin {self.name} loaded")
        self.setup_resources()
    
    def on_start(self):
        """Called when plugin is started"""
        self.subscribe_to_message("app.event", self.handle_event)
    
    def on_stop(self):
        """Called when plugin is stopped"""
        self.cleanup()
    
    def on_unload(self):
        """Called when plugin is unloaded"""
        print(f"Plugin {self.name} unloaded")
    
    def handle_event(self, event_type, **data):
        # Plugin logic
        pass
```

### Loading Plugins

```python
from SLM.core import plugins

# Load from file
plugins.load_plugin("path/to/plugin.py")

# Load from directory
plugins.load_plugins_from_directory("plugins/")

# Load by name (if registered)
plugins.load_plugin_by_name("MyPlugin")
```

### Plugin Management

```python
# Start/stop plugins
plugins.start_plugins()
plugins.stop_plugins()

# Enable/disable specific plugin
plugins.enable_plugin("MyPlugin")
plugins.disable_plugin("MyPlugin")

# Get plugin info
status = plugins.get_plugin_status()
count = plugins.get_plugin_count()
```

---

## Decorators

### Complete Decorator Reference

**File:** `decorators.py`

### @component

Register a class as a component:

```python
@component(name="MyService", auto_register=True)
class MyService(Component):
    pass
```

**Parameters:**
- `name` (optional): Component name
- `auto_register` (default: True): Auto-register with ComponentManager

### @service

Register a class as a service:

```python
@service(singleton=True, name="AuthService")
class AuthService:
    pass
```

**Parameters:**
- `singleton` (default: True): Register as singleton
- `name` (optional): Service name

### @inject

Inject specific dependencies:

```python
@inject(config=Config, bus=MessageBus)
def process(data, config, bus):
    # config and bus are injected
    pass
```

**Parameters:**
- `**dependencies`: Mapping of parameter name to type

### @auto_inject

Automatic injection based on type hints:

```python
@auto_inject
def smart_function(data: str, config: Config):
    # config automatically injected
    pass
```

### @subscribe

Subscribe to events:

```python
@subscribe("user.created", "user.updated")
def on_user_event(event_type, **data):
    print(f"Event: {event_type}")
```

**Parameters:**
- `*events`: Event types to subscribe to

### @on_config_change

React to configuration changes:

```python
@on_config_change("database.host")
def on_db_change(old_value, new_value):
    reconnect(new_value)
```

**Parameters:**
- `key` (optional): Specific config key to watch

### @on_app_start / @on_app_stop

Lifecycle hooks:

```python
@on_app_start
def initialize():
    setup_resources()

@on_app_stop
def cleanup():
    release_resources()
```

### @cached_property

Cache expensive computations:

```python
class DataProcessor:
    @cached_property
    def expensive_result(self):
        # Computed once, then cached
        return compute_expensive_thing()
```

---

## Best Practices

### 1. Configuration Management

```python
# ✅ Good: Use nested config
config.database.primary.host = "db1.example.com"
config.database.replica.host = "db2.example.com"

# ❌ Bad: Flat structure
config.database_primary_host = "db1.example.com"
```

### 2. Event Naming

```python
# ✅ Good: Use dot notation
bus.publish("user.created")
bus.publish("order.payment.completed")

# ❌ Bad: No structure
bus.publish("user_created")
```

### 3. Component Dependencies

```python
# ✅ Good: Declare with type hints
class ServiceA(Component):
    config: Config
    message_bus: MessageBus

# ❌ Bad: Manual injection
class ServiceB(Component):
    def __init__(self):
        self.config = get_config()  # Tight coupling
```

### 4. Error Handling

```python
# ✅ Good: Handle in lifecycle methods
class MyService(Component):
    def on_start(self):
        try:
            self.connect()
        except ConnectionError as e:
            logger.error(f"Failed to connect: {e}")
            raise  # Let framework handle

# ❌ Bad: Swallow errors
class BadService(Component):
    def on_start(self):
        try:
            self.connect()
        except:
            pass  # Silent failure
```

### 5. Testing

```python
# ✅ Good: Reset between tests
def test_config():
    config.value = "test"
    assert config.value == "test"

def test_another():
    reset_all()  # Clean slate
    assert not hasattr(config, 'value')
```

---

## Performance Considerations

### Singleton Overhead

- **Initialization**: O(1) with double-check locking
- **Access**: O(1) dictionary lookup
- **Memory**: Single instance per class

### Event Bus

- **Publish**: O(n) where n = number of subscribers
- **Subscribe**: O(1) list append
- **Queue processing**: Async background thread

### Dependency Injection

- **Setup**: O(n log n) topological sort
- **Injection**: O(d) where d = number of dependencies
- **Cached**: After first injection, O(1) access

---

## Troubleshooting

### Common Issues

#### 1. Circular Dependencies

**Problem:** Services depend on each other
**Solution:** Use events for communication or redesign

```python
# Instead of this:
class A(Component):
    b: B

class B(Component):
    a: A

# Do this:
class A(Component):
    def on_start(self):
        self.send_message("a.ready")

class B(Component):
    def on_initialize(self):
        self.subscribe_to_message("a.ready", self.on_a_ready)
```

#### 2. Config Not Reactive

**Problem:** Config changes don't trigger events
**Solution:** Connect message bus

```python
from SLM.core import config, bus

config.message_bus = bus  # Connect before changes
```

#### 3. Singleton Reset Needed

**Problem:** Tests interfering with each other
**Solution:** Reset in setUp/tearDown

```python
import unittest
from SLM.core import reset_all

class MyTest(unittest.TestCase):
    def setUp(self):
        reset_all()
```

---

## Migration Checklist

- [ ] Replace manual singleton with `Singleton` base class
- [ ] Update config access to use module-level `config`
- [ ] Replace service locator with dependency injection
- [ ] Add `@component` decorator to components
- [ ] Convert event handlers to `@subscribe`
- [ ] Add lifecycle hooks with decorators
- [ ] Update tests to use `reset_all()`
- [ ] Remove manual service registration code

---

**Version:** 2.0.0  
**Last Updated:** 2025-01-11  
**Status:** Complete ✅
