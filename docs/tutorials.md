# USCore Tutorials

## Tutorial 1: Creating a Simple Service

In this tutorial, we will create a `GreetingService` that logs a welcome message.

### Step 1: Define the Service

Create a file `myservices.py`:

```python
from src.core.base_service import BaseService
from loguru import logger

class GreetingService(BaseService):
    def __init__(self):
        super().__init__()

    async def initialize(self):
        await super().initialize()
        logger.info("GreetingService initialized!")

    def greet(self, name: str):
        return f"Hello, {name}! Welcome to USCore."
```

### Step 2: Register and Use

In your `main.py`:

```python
import asyncio
from src.core.locator import sl
from myservices import GreetingService

async def main():
    # 1. Register
    sl.register_service(GreetingService)

    # 2. Initialize all registered services
    await sl.initialize_services()

    # 3. Use
    greeter = sl.get_service(GreetingService)
    print(greeter.greet("Developer"))

if __name__ == "__main__":
    asyncio.run(main())
```

## Tutorial 2: Using the Event Bus

### Step 1: Define an Event Listener

```python
from src.core.events.bus import EventBus
from src.core.locator import sl

async def on_user_login(user_id):
    print(f"User {user_id} logged in!")

async def setup_listeners():
    bus = sl.get_service(EventBus)
    bus.subscribe("user_login", on_user_login)
```

### Step 2: Emit an Event

```python
async def login_user(user_id):
    bus = sl.get_service(EventBus)
    await bus.emit("user_login", user_id)
```
