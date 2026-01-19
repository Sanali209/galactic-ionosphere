# Bootstrap Architecture

## Overview

The bootstrap system provides a fluent API for configuring and initializing applications with proper dependency management and service registration.

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│         Application Entry Point                 │
│  (main.py, CLI tool, worker script)            │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│         ApplicationBuilder                       │
│  .for_console() / .for_gui() / .for_engine()   │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│         System Bundles                          │
├─────────────────────────────────────────────────┤
│  CoreBundle (built-in)                          │
│    • DatabaseManager, TaskSystem, etc.          │
│                                                 │
│  UCoreFSDataBundle                              │
│    • FSService, TagManager, SearchService       │
│                                                 │
│  PySideBundle (optional)                        │
│    • DockingService, SessionState               │
│                                                 │
│  UCoreFSEngineBundle                            │
│    • ProcessingPipeline, AI Extractors          │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│         ServiceLocator                          │
│  - Manages system lifetime                      │
│  - Provides dependency injection                │
└─────────────────────────────────────────────────┘
```

## Core Concepts

### ApplicationBuilder

Fluent API for constructing applications.

**Methods**:
- `for_console(name, config)` - Console application preset
- `for_gui(name, config)` - GUI application preset
- `for_engine(name, config)` - Headless worker preset
- `with_default_systems()` - Add core systems
- `add_system(SystemClass)` - Register single system
- `add_bundle(bundle)` - Register bundle of systems
- `build()` - Initialize and return ServiceLocator

### SystemBundle

Groups related systems for reusability.

**Interface**:
```python
class MyBundle(SystemBundle):
    def register(self, builder: ApplicationBuilder):
        builder.add_system(ServiceA)
        builder.add_system(ServiceB)
```

### ServiceLocator

Manages system registration and lifetime.

**Methods**:
- `register_system(SystemClass)` - Register system
- `get_system(SystemClass)` - Retrieve system instance
- `start_all()` - Initialize all systems
- `stop_all()` - Shutdown all systems

## Bundle Hierarchy

### Level 1: Core (Foundation)

**Built-in via `.with_default_systems()`**:
- DatabaseManager
- CommandBus
- JournalService
- AssetManager
- TaskSystem
- PeriodicTaskScheduler

### Level 2: Domain (UCoreFS)

**UCoreFSDataBundle** - Framework-agnostic data:
- FSService - File system metadata
- TagManager - Tag hierarchy
- AlbumManager - Album management
- RelationService - File relations
- SearchService - Query engine
- VectorService - Embeddings
- FAISSIndexService - Vector search
- ThumbnailService - Thumbnail generation

### Level 3: UI (Optional)

**PySideBundle** - Qt framework:
- DockingService - Panel management
- SessionState - UI state persistence
- NavigationService - Selection routing

### Level 4: Processing (Optional)

**UCoreFSEngineBundle** - AI/Processing:
- ProcessingPipeline - File processing orchestration
- AI Extractors - CLIP, BLIP, GroundingDINO
- WDTaggerService - Image tagging
- DetectionService - Object detection
- MaintenanceService - Automated tasks

## Application Types

### Console Application

**Characteristics**:
- No GUI dependencies
- Fast startup
- Headless operation
- CLI interface

**Bundle Stack**:
```
CoreBundle (built-in)
  └─ UCoreFSDataBundle
```

**Example**:
```python
locator = await (ApplicationBuilder.for_console("CLI")
    .add_bundle(UCoreFSDataBundle())
    .build())
```

### GUI Application

**Characteristics**:
- Full UI framework
- User interaction
- Window management

**Bundle Stack**:
```
CoreBundle (built-in)
  ├─ UCoreFSDataBundle
  ├─ PySideBundle
  └─ AppUIBundle
```

**Example**:
```python
builder = (ApplicationBuilder.for_gui("UExplorer")
    .add_bundle(UCoreFSDataBundle())
    .add_bundle(PySideBundle())
    .add_bundle(UExplorerUIBundle()))
```

### Processing Engine

**Characteristics**:
- Background processing
- No UI
- AI/ML workloads

**Bundle Stack**:
```
CoreBundle (built-in)
  ├─ UCoreFSDataBundle
  └─ UCoreFSEngineBundle
```

**Example**:
```python
locator = await (ApplicationBuilder.for_engine("Engine")
    .add_bundle(UCoreFSDataBundle())
    .add_bundle(UCoreFSEngineBundle())
    .build())
```

## Dependency Flow

```
ApplicationBuilder.for_console()
    ↓
  .with_default_systems()  → CoreBundle
    ↓
  .add_bundle(DataBundle)  → UCoreFSDataBundle
    ↓
  .build()
    ↓
  ServiceLocator
    ↓
  await sl.start_all()
    ↓
  Systems Initialized
```

## Extension Points

### Creating Custom Bundles

```python
class MyAppBundle(SystemBundle):
    def register(self, builder: ApplicationBuilder):
        builder.add_system(MyService)
        builder.add_system(MyWorker)

# Usage
builder.add_bundle(MyAppBundle())
```

### Adding Systems

```python
builder.add_system(CustomService)
```

### Conditional Registration

```python
if feature_enabled:
    builder.add_system(FeatureService)
```

## Best Practices

1. **Use Presets** - Prefer `.for_console()` over manual config
2. **Bundle Related Services** - Group dependencies together
3. **Respect Dependencies** - Register in correct order
4. **Validate Requirements** - Check prerequisites in bundle.register()
5. **Document Bundles** - Clear docstrings for each bundle

## Performance

### Startup Time

- **Console**: ~1s (no Qt)
- **GUI**: ~3s (with Qt + UI init)
- **Engine**: ~2s (with AI models)

### Memory

- **Console**: ~50MB
- **GUI**: ~200MB (Qt framework)
- **Engine**: ~500MB+ (AI models)

## Security

- Configuration via `config.json`
- Secrets via `.env` file
- No hardcoded credentials
- Service isolation via locator

## Testing

### Unit Tests

```python
async def test_console_builder():
    locator = await (ApplicationBuilder.for_console("Test")
        .add_bundle(UCoreFSDataBundle())
        .build())
    
    assert locator is not None
    await locator.stop_all()
```

### Integration Tests

```python
async def test_data_bundle():
    locator = await build_test_app()
    fs = locator.get_system(FSService)
    assert fs is not None
```
