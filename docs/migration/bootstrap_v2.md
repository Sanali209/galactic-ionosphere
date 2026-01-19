# Bootstrap Architecture Migration Guide

## Overview

This guide explains the new bootstrap architecture and how to migrate existing code.

## What Changed?

### 1. PySide6 is Now Optional

**Before**: PySide6 was always imported, even for console apps  
**After**: PySide6 only imported when using `run_app()` or `PySideBundle`

### 2. New Builder Presets

**Added**:
- `ApplicationBuilder.for_console()` - Console applications
- `ApplicationBuilder.for_gui()` - GUI applications  
- `ApplicationBuilder.for_engine()` - Headless processing

### 3. New Bundles

**Added**:
- `PySideBundle` - PySide6 UI framework services
- `UCoreFSDataBundle` - Framework-agnostic data layer

**Renamed**:
- `UCoreFSClientBundle` → `UCoreFSDataBundle` (alias exists)

---

## Migration Examples

### Console Application

**Before** (didn't work):
```python
from src.core.bootstrap import ApplicationBuilder  # ❌ Imported PySide6!
```

**After** (works!):
```python
from src.core.bootstrap import ApplicationBuilder

locator = await (ApplicationBuilder.for_console("MyCLI")
    .add_bundle(UCoreFSDataBundle())
    .build())
```

### GUI Application

**Before**:
```python
builder = (ApplicationBuilder("MyApp", "config.json")
    .with_default_systems()
    .with_logging(True)
    .add_bundle(UExplorerUIBundle())
    .add_bundle(UCoreFSClientBundle()))
```

**After** (cleaner):
```python
builder = (ApplicationBuilder.for_gui("MyApp", "config.json")
    .add_bundle(UCoreFSDataBundle())
    .add_bundle(PySideBundle())
    .add_bundle(UExplorerUIBundle()))
```

### Engine/Worker

**Before**:
```python
builder = ApplicationBuilder("Engine", "config.json")
builder.with_default_systems(False)
builder.add_system(DatabaseManager)
builder.add_system(CommandBus)
# ... 5 more lines
builder.add_bundle(UCoreFSEngineBundle())
```

**After** (2 lines!):
```python
builder = (ApplicationBuilder.for_engine("Engine", "config.json")
    .add_bundle(UCoreFSEngineBundle()))
```

---

## Backward Compatibility

All old code continues to work:

```python
# Still works!
builder = ApplicationBuilder("App").with_default_systems()

# Still works!
from src.ucorefs.bundle import UCoreFSClientBundle
bundle = UCoreFSClientBundle()  # Uses alias to UCoreFSDataBundle
```

---

## Bundle Reference

### CoreBundle (Built-in)

Included via `.with_default_systems()` or `.for_*()` presets:
- DatabaseManager
- CommandBus
- JournalService
- AssetManager
- TaskSystem
- PeriodicTaskScheduler

### UCoreFSDataBundle

Framework-agnostic data layer:
- FSService
- TagManager
- AlbumManager
- RelationService
- SearchService
- VectorService
- FAISSIndexService
- ThumbnailService

**Usage**:
```python
from src.ucorefs.bundles import UCoreFSDataBundle

# Read-write
.add_bundle(UCoreFSDataBundle())

# Read-only
.add_bundle(UCoreFSDataBundle(readonly=True))
```

### PySideBundle

Qt/PySide6 UI framework:
- DockingService
- SessionState
- NavigationService

**Usage**:
```python
from src.ui.pyside_bundle import PySideBundle

.add_bundle(PySideBundle(theme="dark"))
```

**Requires**: `pip install PySide6`

### UCoreFSEngineBundle

Processing/AI layer:
- ProcessingPipeline
- AI Extractors (CLIP, BLIP, etc.)
- WDTaggerService
- DetectionService
- MaintenanceService

**Usage**:
```python
from src.ucorefs.bundle import UCoreFSEngineBundle

.add_bundle(UCoreFSEngineBundle())
```

---

## Common Patterns

### Console Tool

```python
#!/usr/bin/env python3
import asyncio
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.bundles import UCoreFSDataBundle

async def main():
    locator = await (ApplicationBuilder.for_console("MyTool")
        .add_bundle(UCoreFSDataBundle())
        .build())
    
    # Use services
    from src.ucorefs.services.fs_service import FSService
    fs = locator.get_system(FSService)
    
    # ...
    
    await locator.stop_all()

asyncio.run(main())
```

### GUI Application

```python
from src.core.bootstrap import ApplicationBuilder, run_app
from src.ucorefs.bundles import UCoreFSDataBundle
from src.ui.pyside_bundle import PySideBundle

builder = (ApplicationBuilder.for_gui("MyApp")
    .add_bundle(UCoreFSDataBundle())
    .add_bundle(PySideBundle())
    .add_bundle(MyUIBundle()))

run_app(MainWindow, MainViewModel, builder=builder)
```

### Headless Server

```python
import asyncio
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.bundles import UCoreFSDataBundle
from src.ucorefs.bundle import UCoreFSEngineBundle

async def main():
    locator = await (ApplicationBuilder.for_engine("Server")
        .add_bundle(UCoreFSDataBundle())
        .add_bundle(UCoreFSEngineBundle())
        .build())
    
    # Start processing
    task_system = locator.get_system(TaskSystem)
    await task_system.start_workers()
    
    # Keep running
    await asyncio.Event().wait()

asyncio.run(main())
```

---

## Troubleshooting

**Error: "No module named 'PySide6'"**

If using console app:
✅ This is expected! Console apps don't need PySide6.

If using GUI app:
```bash
pip install PySide6 qasync
```

**Error: "PySideBundle requires PySide6"**

Remove `PySideBundle` if building console app:
```python
# Console - no PySideBundle
builder = ApplicationBuilder.for_console("CLI").build()
```

**Old code stopped working**

Check if you're using deprecated names:
- `UCoreFSClientBundle` → Use `UCoreFSDataBundle` instead

---

## Deprecation Timeline

- **Now**: Aliases work, warnings logged
- **v2.0**: Aliases removed

Use `UCoreFSDataBundle` for future compatibility.
