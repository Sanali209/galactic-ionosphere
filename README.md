# USCore - Universal System Core

**A professional Python desktop application framework with async support, PySide6 GUI, and MongoDB integration.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/PySide6-Qt6-green.svg)](https://www.qt.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-ODM-brightgreen.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

USCore provides a **production-ready foundation** for building desktop applications with Python. It eliminates months of boilerplate development by providing:

- ✅ **Bootstrap Helpers** - 75% less code to start an app
- ✅ **Async MongoDB ORM** - Auto-collection naming, relationships, reactive models
- ✅ **Advanced GUI Framework** - Docking panels, command palette, shortcuts
- ✅ **MVVM Architecture** - Clean separation of concerns
- ✅ **Task System** - Background processing with auto-recovery
- ✅ **Service Locator** - Dependency injection and lifecycle management

## 📦 Repository Structure

```
USCore/
├── templates/
│   └── foundation/          # Core framework template
│       ├── src/             # Framework source code
│       │   ├── core/        # Core systems (ORM, Tasks, Config, etc.)
│       │   └── ui/          # GUI framework (Docking, MVVM, Menus)
│       ├── docs/            # Comprehensive documentation
│       ├── tests/           # 56 unit tests (85%+ coverage)
│       ├── pyproject.toml   # pip-installable package
│       └── README.md        # Foundation documentation
│
└── samples/
    └── image_search/        # Example: DuckDuckGo image search app
        ├── main.py          # 33 lines (vs 130 before!)
        └── src/             # App-specific code
```

## 🚀 Quick Start

### 1. Install Foundation Package

```bash
cd templates/foundation
pip install -e .
```

### 2. Create Your First App

```python
# main.py
from foundation import ApplicationBuilder, run_app
from src.ui.main_window import MainWindow
from src.ui.viewmodels.main_viewmodel import MainViewModel

if __name__ == "__main__":
    builder = (ApplicationBuilder("My App", "config.json")
               .with_default_systems()
               .with_logging(True))
    
    run_app(MainWindow, MainViewModel, builder=builder)
```

### 3. Define Your Data Model

```python
from foundation import CollectionRecord, Field

class User(CollectionRecord):
    # Auto collection name: "users"
    username: str = Field(default="", index=True, unique=True)
    email: str = Field(default="")
    role: str = Field(default="user")

# Use it
user = User(username="alice", email="alice@example.com")
await user.save()
```

**That's it!** You have a fully functional async desktop app with:
- MongoDB integration
- Background task processing
- Logging system
- Configuration management
- Professional GUI framework

## ✨ Key Features

### Foundation Template v0.1.0

#### 🔧 Bootstrap Pattern (New!)
- **ApplicationBuilder** - Fluent configuration API
- **run_app()** - One-liner app execution
- **75% code reduction** - From 130 to 33 lines in main.py

#### 🗄️ Async MongoDB ORM
- **Auto-collection naming** - `ImageRecord` → `"image_records"`
- **Smart pluralization** - `SearchHistory` → `"search_histories"`
- **Full Field() control** - Index, unique, validators
- **Relationships** - 1:1, 1:N, N:N with lazy loading
- **Reactive changes** - ObserverEvent integration

#### 🎨 Advanced GUI Framework
- **Dock Management** - Resizable panels with state save
- **Command Palette** - Ctrl+Shift+P fuzzy search
- **30+ Keyboard Shortcuts** - Full navigation
- **Settings Dialog** - Ctrl+, for preferences
- **Document Splits** - Unlimited nested layouts

#### ⚙️ Core Systems
- **ServiceLocator** - Dependency injection
- **TaskSystem** - Background processing with MongoDB queue
- **ConfigManager** - Pydantic-backed reactive configuration
- **JournalService** - Structured logging to database
- **CommandBus** - Decoupled command/handler pattern
- **AssetManager** - Extensible file handling

## 📚 Documentation

### Foundation Template
- **[Getting Started](templates/foundation/docs/guides/getting_started.md)** - Installation & first app
- **[Migration Guide](templates/foundation/MIGRATION.md)** - Upgrade from old patterns
- **[ORM Guide](templates/foundation/docs/components/orm.md)** - Database models & queries
- **[Systems Architecture](templates/foundation/docs/architecture/systems.md)** - Framework internals
- **[Full Documentation](templates/foundation/docs/index.md)** - Complete reference

### Quick Links
- [Current State](templates/foundation/docs/current_state.md) - v0.1.0 features
- [GUI User Guide](templates/foundation/docs/guides/gui_user_guide.md) - Shortcuts & tips
- [API Reference](templates/foundation/docs/api/gui_framework.md) - Code examples

## 🎯 Sample Applications

### Image Search (DuckDuckGo)
**Location**: `samples/image_search/`

A complete desktop app demonstrating foundation features:
- DuckDuckGo API integration
- MongoDB data persistence
- Custom BaseSystem (SearchService)
- ORM models with relationships
- Background task processing
- Advanced UI with panels

**Run it:**
```bash
cd samples/image_search
python main.py
```

**Code reduction**: 130 lines → **33 lines** (75% less!)

## 🛠️ Development Setup

### Prerequisites
- Python 3.10+
- MongoDB (running locally or accessible)
- pip

### Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd USCore

# Install foundation in development mode
cd templates/foundation
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
# From foundation directory
pytest tests/ -v

# With coverage
pytest --cov=src --cov-report=html
```

**Test Stats**: 56 tests | 85%+ coverage

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Production Code | 1,600+ lines |
| Bootstrap Infrastructure | ~400 lines |
| Unit Tests | 56 tests |
| Test Coverage | 85%+ |
| Modules | 19 modules |
| Subsystems | 5 core systems |

## 🎓 Examples

### Creating a Custom System

```python
from foundation import BaseSystem
from loguru import logger

class NotificationService(BaseSystem):
    async def initialize(self):
        logger.info("NotificationService starting")
        self.channels = []
        await super().initialize()
    
    async def send(self, message: str):
        # Your notification logic
        pass
    
    async def shutdown(self):
        self.channels.clear()
        await super().shutdown()

# Register it
builder = (ApplicationBuilder("My App")
           .with_default_systems()
           .add_system(NotificationService))
```

### ORM with Relationships

```python
from foundation import CollectionRecord, Field, ReferenceField

class Author(CollectionRecord):
    # Auto collection: "authors"
    name: str = Field(default="")
    email: str = Field(default="", index=True, unique=True)

class BlogPost(CollectionRecord):
    # Auto collection: "blog_posts"
    title: str = Field(default="", index=True)
    content: str = Field(default="")
    author: Author = ReferenceField(Author)

# Create and save
author = Author(name="Alice", email="alice@example.com")
await author.save()

post = BlogPost(title="Hello World", author=author)
await post.save()

# Query and resolve
post = await BlogPost.find_one({"title": "Hello World"})
author = await post.author.fetch()  # Lazy load
print(author.name)  # "Alice"
```

## 🔄 Migration from Old Pattern

If you have existing code using the old pattern:

**Before** (130 lines):
```python
# Complex imports, path hacks
import importlib.util
sys.path.insert(0, ...)

# Manual system registration
sl = ServiceLocator()
sl.init()
sl.register_system(DatabaseManager)
# ... 10+ more lines

# Qt/async boilerplate
app = QApplication(sys.argv)
loop = QEventLoop(app)
# ... 20+ more lines
```

**After** (33 lines):
```python
from foundation import ApplicationBuilder, run_app

builder = (ApplicationBuilder("My App")
           .with_default_systems())

run_app(MainWindow, MainViewModel, builder=builder)
```

See **[MIGRATION.md](templates/foundation/MIGRATION.md)** for complete guide.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Application (main.py)           │
│    ApplicationBuilder + run_app()       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Service Locator                 │
│  Dependency Injection & Lifecycle       │
└─────────┬───────────────────────────────┘
          │
    ┌─────┴─────────────────────────┐
    │                               │
┌───▼────────────┐      ┌──────────▼──────┐
│  Core Systems  │      │   UI Framework  │
│  - Database    │      │   - Docking     │
│  - Tasks       │      │   - MVVM        │
│  - Config      │      │   - Menus       │
│  - Journal     │      │   - Commands    │
│  - Commands    │      │   - Settings    │
└────────────────┘      └─────────────────┘
```

## 🤝 Contributing

This is a template repository. To use it:

1. **Use the foundation** as a starting point for your project
2. **Extend systems** with your custom implementations
3. **Share improvements** back to the template

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- [PySide6](https://www.qt.io/) - Qt6 Python bindings
- [Motor](https://motor.readthedocs.io/) - Async MongoDB driver
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Loguru](https://loguru.readthedocs.io/) - Logging
- [qasync](https://github.com/CabbageDevelopment/qasync) - Qt async integration

---

**Ready to build your next desktop application?** Start with USCore and focus on what makes your app unique, not on plumbing! 🚀
