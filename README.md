# USCore Foundation

Professional Python application framework for PySide6 desktop applications.

## Features

- **ServiceLocator**: Dependency injection container with lifecycle management
- **CommandBus**: Decoupled command pattern implementation
- **TaskSystem**: Async background task processing
- **DockingService**: Professional dockable panel system (PySide6-QtAds)
- **UCoreFS**: Filesystem database with AI capabilities

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.core.bootstrap import ApplicationBuilder, run_app
from src.ui.main_window import MainWindow

builder = ApplicationBuilder("MyApp").with_default_systems()
run_app(MainWindow, builder=builder)
```

## Samples

- **UExplorer**: File manager showcasing Foundation + UCoreFS
- **Node Editor**: Visual node programming environment
- **Image Search**: AI-powered image search

## License

MIT
