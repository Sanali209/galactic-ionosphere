# Adding a New Feature

This guide walks through adding a new "Notification System" to the app.

## Step 1: Define the System
Create `src/core/notifications/system.py`.

```python
from ..base_system import BaseSystem

class NotificationSystem(BaseSystem):
    async def initialize(self):
        print("Notifications ready")
        await super().initialize()
        
    def send(self, msg):
        print(f"ALERT: {msg}")
```

## Step 2: Register in Locator
Update `src/main.py`.

```python
from src.core.notifications.system import NotificationSystem

# ...
sl.register_system(NotificationSystem)
```

## Step 3: Use in UI
Update `src/ui/bridge.py` or a Widget.

```python
# In a widget
notif_sys = self.locator.get_system(NotificationSystem)
notif_sys.send("Hello World")
```

## Best Practices
- **Use Commands**: If the action modifies state, create a Command for it.
- **Use Events**: If other parts of the system need to know, emit an Event.
- **Log Everything**: Use `src.core.logging`.
