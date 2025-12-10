import asyncio
import sys
from src.core.locator import sl

from src.ui.main_window import MainWindow
from src.core.capabilities.base import IDriver
from src.core.database.manager import db_manager
from src.core.database.orm import CollectionRecord, FieldPropInfo

# --- Dummy Driver ---
class MockVectorDriver(IDriver):
    def __init__(self, driver_id):
        self._id = driver_id
    @property
    def id(self) -> str: return self._id
    def load(self): print(f"[Driver] Loaded: {self.id}")
    def unload(self): print(f"[Driver] Unloaded: {self.id}")

# --- Test Records ---
class UserRecord(CollectionRecord, table="users"):
    name = FieldPropInfo("name", default="Guest", field_type=str)
    age = FieldPropInfo("age", default=0, field_type=int)

class AdminRecord(UserRecord): # Polymorphism test
    role = FieldPropInfo("role", default="admin", field_type=str)

async def async_main():
    print("--- 1. Initialize Core ---")
    sl.init("settings.json")
    
    # Initialize DB (Async)
    print("--- 1b. Initialize Mongo ---")
    db_manager.init()
    
    print("--- 2. Register Drivers ---")
    sl.caps.ai_vectors.register(MockVectorDriver("clip_local"))
    sl.caps.ai_vectors.register(MockVectorDriver("gpt4"))
    sl.caps.ai_vectors.switch("clip_local")

    print("--- 3. ORM Test: Create & Save ---")
    user = UserRecord(name="Alice", age=30)
    print(f"Created User: {user.name}, Age: {user.age}")
    
    # Reactive Event Test
    def on_user_change(obj, field, val):
        print(f"[Event] User changed '{field}' to '{val}'")
    user.on_change.connect(on_user_change)
    
    user.age = 31 # Should trigger event
    
    try:
        await user.save()
        print(f"Saved User ID: {user.id}")
    except Exception as e:
        print(f"Save failed (Expected if no DB connection): {e}")

    print("--- 4. ORM Test: Polymorphism ---")
    admin = AdminRecord(name="Bob", age=40)
    print(f"Created Admin: {admin.name}, Role: {admin.role}")
    try:
        await admin.save()
        print(f"Saved Admin ID: {admin.id}")
        
        # Load back to check polymorphism
        loaded_admin = await UserRecord.get(admin.id)
        if loaded_admin:
            print(f"Loaded back as: {type(loaded_admin).__name__}")
            print(f"Data: {loaded_admin.name}, {loaded_admin.role}")
    except Exception as e:
         print(f"Polymorphism Save/Load failed: {e}")

    print("--- 5. Cleanup ---")
    # In a real app we'd close the client, but motor client handles itself well.

def main():
    # Run async loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_main())
    loop.close()

if __name__ == "__main__":
    main()
