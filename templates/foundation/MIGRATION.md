# Foundation Template Migration Guide

This guide shows how to upgrade existing samples to use the new Foundation v0.1.0 improvements.

## Installation

### Step 1: Install Foundation Package

```bash
cd templates/foundation
pip install -e .
```

After installation, you can import from `foundation` package instead of using `sys.path.insert` hacks.

---

## Migration Path

### Before: Old Pattern (130 lines)

```python
# main.py
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent
FOUNDATION_DIR = APP_DIR.parent.parent / "templates/foundation"
sys.path.insert(0, str(FOUNDATION_DIR))

# 40+ lines of importlib.util hacks...

async def main():
    setup_logging()
    sl = ServiceLocator()
    sl.init(config_path)
    
    # Manual system registration (5-10 lines)
    sl.register_system(DatabaseManager)
    sl.register_system(CommandBus)
    # ... more systems
    
    await sl.start_all()
    
    # Create window
    main_vm = MainViewModel(sl)
    window = MainWindow(main_vm)
    window.show()
    return sl

if __name__ == "__main__":
    # Qt/async boilerplate (20+ lines)
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    # ... event loop setup
```

### After: New Pattern (33 lines)

```python
# main.py
from foundation import ApplicationBuilder, run_app
from src.core.search_service import SearchService
from src.ui.viewmodels.main_viewmodel import MainViewModel
from src.ui.main_window import MainWindow

if __name__ == "__main__":
    builder = (ApplicationBuilder("My App", "config.json")
               .with_default_systems()
               .with_logging(True)
               .add_system(SearchService))
    
    run_app(MainWindow, MainViewModel, builder=builder)
```

**Result**: **75% reduction** (130 → 33 lines)

---

## Detailed Changes

### 1. Update Models

**Before**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "templates/foundation"))

from src.core.database.orm import CollectionRecord, Field

class ImageRecord(CollectionRecord):
    _collection_name = "images"
    url: str = Field(default="")
```

**After**:
```python
from foundation import CollectionRecord, Field

class ImageRecord(CollectionRecord):
    # Auto collection name: "image_records"
    url: str = Field(default="")
```

**Changes**:
- ✅ Remove `sys.path.insert` (4 lines)
- ✅ Use `from foundation import ...`
- ✅ Remove `_collection_name` (auto-generated)

---

### 2. Update Services

**Before**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent... / "foundation"))

from src.core.base_system import BaseSystem

class MyService(BaseSystem):
    async def initialize(self):
        ...
```

**After**:
```python
from foundation import BaseSystem

class MyService(BaseSystem):
    async def initialize(self):
        ...
```

**Changes**:
- ✅ Remove `sys.path.insert`
- ✅ Use `from foundation import BaseSystem`

---

### 3. Update UI

**Before**:
```python
sys.path.insert(0, str(Path(__file__)... / "foundation"))

from src.ui.docking.dock_manager import DockManager
from src.ui.menus.action_registry import ActionRegistry
```

**After**:
```python
from foundation import DockManager, ActionRegistry, MenuBuilder
```

---

## Auto-Collection Naming

The ORM now automatically generates collection names from class names:

| Class Name | Auto Collection Name |
|-----------|---------------------|
| `ImageRecord` | `image_records` |
| `SearchHistory` | `search_histories` |
| `User` | `users` |
| `HTTPResponse` | `http_responses` |

### Override if needed:
```python
class ImageRecord(CollectionRecord):
    _collection_name = "my_custom_images"  # Override auto-naming
```

---

## Bootstrap API

### ApplicationBuilder

```python
builder = (ApplicationBuilder("App Name", "config.json")
           .with_default_systems()      # Includes DB, CommandBus, etc.
           .with_logging(True)           # Setup logging
           .add_system(MyService))       # Add custom system
```

### run_app

```python
run_app(MainWindowClass, ViewModelClass, builder=builder)
```

Handles:
- Qt application setup
- Event loop management
- Async initialization
- Window creation
- Graceful shutdown

---

## Verification

After migrating, verify:

1. **Import works without errors**:
```bash
python -c "from foundation import ApplicationBuilder, CollectionRecord"
```

2. **Application starts**:
```bash
python main.py
```

3. **Database collections match** (check MongoDB):
- Old: `images`, `search_history`
- New: `image_records`, `search_histories`

**Note**: You may need to migrate data or use explicit `_collection_name` to maintain compatibility.

---

## Checklist

- [ ] Install foundation: `pip install -e templates/foundation`
- [ ] Update `main.py` to use `ApplicationBuilder` and `run_app`
- [ ] Remove all `sys.path.insert` from model files
- [ ] Update imports to `from foundation import ...`
- [ ] Remove explicit `_collection_name` (or keep for compatibility)
- [ ] Test application startup
- [ ] Verify database connections
- [ ] Check collection names in MongoDB

---

## Need Help?

See `templates/foundation/README.md` for full API documentation.
