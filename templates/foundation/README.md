# Foundation Template

Professional desktop application framework with async Python, PySide6, MongoDB, and advanced GUI.

## Quick Start

```bash
# Install foundation as package (development mode)
cd templates/foundation
pip install -e .

# Run sample application
cd ../../samples/image_search
python main.py
```

## Features

### Core Architecture ✅
- **ServiceLocator** - Dependency injection
- **AsyncIO** - Full async/await with qasync
- **MVVM Pattern** - Clean separation of concerns
- **ConfigManager** - Pydantic reactive config
- **ORM** - MongoDB with 1:1, 1:N, N:N relationships (auto-collection naming)
- **Task System** - Background processing with recovery
- **CommandBus** - Decoupled command execution
- **Bootstrap Helpers** - ApplicationBuilder & run_app for minimal boilerplate

### GUI Framework ✅
- **Document Splits** - Unlimited layouts
- **Docking Panels** - Resizable with state save
- **Complete Menus** - All standard actions
- **Settings Dialog** - Ctrl+, for preferences  
- **Command Palette** - Ctrl+Shift+P fuzzy search
- **30+ Shortcuts** - Full keyboard navigation

## Documentation

- **[User Guide](./docs/guides/gui_user_guide.md)** - Features and shortcuts
- **[Developer Guide](./docs/guides/gui_developer_guide.md)** - Extending the framework
- **[API Reference](./docs/api/gui_framework.md)** - Code examples
- **[Full Docs](./docs/index.md)** - Complete documentation

## Project Structure

```
templates/foundation/
├── main.py                 # Application entry point
├── src/
│   ├── core/              # Core systems
│   │   ├── locator.py     # Service locator
│   │   ├── config.py      # Configuration
│   │   ├── database/      # ORM and DB
│   │   └── tasks/         # Task system
│   └── ui/                # GUI framework
│       ├── documents/     # Document splits
│       ├── docking/       # Panel management
│       ├── menus/         # Menu/action system
│       ├── settings/      # Settings dialog
│       └── commands/      # Command palette
├── tests/                 # Unit tests (56 tests)
└── docs/                  # Documentation
```

## Testing

```bash
# Run all tests
pytest templates/foundation/tests/

# Run specific phase
pytest templates/foundation/tests/test_split_manager.py -v

# With coverage
pytest --cov=src --cov-report=html
```

## Usage Examples

### Using Settings
```bash
# Press Ctrl+,
# Change Application Name
# Changes save automatically
```

### Using Command Palette
```bash
# Press Ctrl+Shift+P
# Type "save" to find save commands
# Press Enter to execute
```

### Creating Custom Panels
```python
from foundation import BasePanelWidget

class MyPanel(BasePanelWidget):
    def initialize_ui(self):
        # Build your UI
        pass
```

### Using Bootstrap (New!) ⚡

```python
from foundation import ApplicationBuilder, run_app
from .ui.main_window import MainWindow
from .ui.viewmodels.main_viewmodel import MainViewModel
from .core.my_service import MyService

# One-liner app setup
builder = (ApplicationBuilder("My App", "config.json")
           .with_default_systems()
           .with_logging(True)
           .add_system(MyService))

run_app(MainWindow, MainViewModel, builder=builder)
```

**⚠️ Migrating from old pattern?** See [MIGRATION.md](./MIGRATION.md)

## Statistics

- **1,600+ lines** production code
- **56 unit tests** (85%+ coverage)
- **19 modules** across 5 subsystems
- **Production ready**

## License

Your license here
