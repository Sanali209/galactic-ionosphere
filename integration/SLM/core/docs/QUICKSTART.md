# SLM Core Quick Start Guide

## Installation

```bash
# Clone or install the package
pip install -e .
```

## 5-Minute Quick Start

### 1. Basic Configuration

```python
from SLM.core import config

# Set configuration values
config.database.host = "localhost"
config.database.port = 5432
config.app.name = "MyApp"
config.app.debug = True

# Access values
print(f"Connecting to {config.database.host}:{config.database.port}")
```

### 2. Publish and Subscribe to Events

```python
from SLM.core import bus, subscribe

# Subscribe to events with decorator
@subscribe("user.created")
def on_user_created(event_type, **data):
    print(f"User {data['username']} was created!")

# Publish an event
bus.publish("user.created", user_id=123, username="john")
```

### 3. Create a Component

```python
from SLM.core import Component, component, Config, MessageBus

@component(name="EmailService")
class EmailService(Component):
    # Dependencies are auto-injected
    config: Config
    message_bus: MessageBus
    
    def on_start(self):
        print(f"Email service started for {self.config.app.name}")
        self.subscribe_to_message("user.created", self.send_welcome_email)
    
    def send_welcome_email(self, event_type, **data):
        print(f"Sending welcome email to {data['username']}")
```

### 4. Register and Run

```python
from SLM.core import dependencies, components

# Create component
email_service = EmailService()

# Register with dependency manager
dependencies.register_singleton(EmailService, email_service)
dependencies.register_singleton(Config, config)
dependencies.register_singleton(MessageBus, bus)

# Setup dependencies (inject + initialize)
dependencies.setup_dependencies()

# Register with component manager
components.register_component("email", email_service)

# Start the component
components.start_all()

# Trigger event - component will handle it
bus.publish("user.created", user_id=456, username="jane")
```

## Common Patterns

### Pattern 1: Simple Service

```python
from SLM.core import service

@service(singleton=True)
class AuthService:
    def __init__(self):
        self.sessions = {}
    
    def login(self, username, password):
        # Authentication logic
        return True
```

### Pattern 2: Dependency Injection

```python
from SLM.core import inject, Config, MessageBus

@inject(config=Config, bus=MessageBus)
def process_order(order_data, config, bus):
    # Process order
    bus.publish("order.processed", order_id=order_data['id'])
    return f"Processed order for {config.app.name}"
```

### Pattern 3: Configuration Changes

```python
from SLM.core import on_config_change

@on_config_change("database.host")
def on_db_host_changed(old_value, new_value):
    print(f"Reconnecting: {old_value} → {new_value}")
    reconnect_database(new_value)

# When config changes, handler is called
config.database.host = "new-server.com"
```

### Pattern 4: Lifecycle Hooks

```python
from SLM.core import on_app_start, on_app_stop

@on_app_start
def initialize_resources():
    print("Setting up resources...")
    # Initialize database, cache, etc.

@on_app_stop
def cleanup_resources():
    print("Cleaning up...")
    # Close connections, save state, etc.
```

## Complete Example

Here's a complete working example:

```python
from SLM.core import (
    config, bus, dependencies, components,
    Component, component, subscribe,
    on_app_start, on_app_stop,
    Config, MessageBus
)

# Configure
config.app.name = "MyApp"
config.database.host = "localhost"

# Lifecycle hooks
@on_app_start
def startup():
    print("App starting...")

@on_app_stop
def shutdown():
    print("App stopping...")

# Define components
@component(name="UserService")
class UserService(Component):
    config: Config
    message_bus: MessageBus
    
    def on_start(self):
        print("User service started")
        self.subscribe_to_message("user.create", self.create_user)
    
    def create_user(self, event_type, **data):
        username = data['username']
        print(f"Creating user: {username}")
        # Create user in database
        self.message_bus.publish("user.created", username=username)

@component(name="EmailService")
class EmailService(Component):
    def on_start(self):
        print("Email service started")
        self.subscribe_to_message("user.created", self.send_email)
    
    def send_email(self, event_type, **data):
        print(f"Sending welcome email to {data['username']}")

# Setup
def setup():
    # Create components
    user_service = UserService()
    email_service = EmailService()
    
    # Register
    dependencies.register_singleton(UserService, user_service)
    dependencies.register_singleton(EmailService, email_service)
    dependencies.register_singleton(Config, config)
    dependencies.register_singleton(MessageBus, bus)
    
    # Setup dependencies
    dependencies.setup_dependencies()
    
    # Register with component manager
    components.register_component("users", user_service)
    components.register_component("email", email_service)
    
    # Start
    components.start_all()

# Run
if __name__ == "__main__":
    setup()
    
    # Trigger events
    bus.publish("user.create", username="john")
    bus.publish("user.create", username="jane")
    
    # Cleanup
    components.stop_all()
```

## Next Steps

1. Read the [Architecture Guide](ARCHITECTURE.md) for detailed concepts
2. Check [API Reference](API_REFERENCE.md) for complete API docs
3. See [examples/elegant_usage_example.py](../examples/elegant_usage_example.py) for more patterns
4. Review [todo.md](../../todo.md) for implementation details

## Tips

### Testing

```python
from SLM.core import reset_all

def test_something():
    # Use singletons
    config.value = "test"
    
    # Your test
    assert config.value == "test"
    
    # Clean up
    reset_all()
```

### Debugging

```python
from loguru import logger

# Enable debug logging
logger.add("debug.log", level="DEBUG")

# Configuration changes are logged
config.database.host = "localhost"  # Logged!
```

### Module-level vs Core Access

```python
# Both work the same:
from SLM.core import config
config.app.name = "MyApp"

from SLM.core import Core
Core.config.app.name = "MyApp"
```

## Common Pitfalls

### 1. Forgetting to Connect Message Bus to Config

```python
# For reactive config, connect message bus first
config.message_bus = bus
config.database.host = "localhost"  # Now triggers events
```

### 2. Not Calling setup_dependencies()

```python
# After registering, must setup
dependencies.register_singleton(MyService, service)
dependencies.setup_dependencies()  # Don't forget!
```

### 3. Circular Dependencies

```python
# ❌ Don't do this
class A(Component):
    b: B

class B(Component):
    a: A

# ✅ Use events instead
class A(Component):
    def on_start(self):
        self.send_message("a.ready")

class B(Component):
    def on_initialize(self):
        self.subscribe_to_message("a.ready", self.handler)
```

## Help & Support

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed docs
- See [examples/](../examples/) for working code
- Read [todo.md](../../todo.md) for roadmap

---

**Version:** 2.0.0  
**Status:** Ready to use ✅
