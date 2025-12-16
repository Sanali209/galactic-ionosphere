# Galactic Ionosphere Foundation Template

A rich, pre-set architecture for building advanced desktop applications with Python, PySide6, and MongoDB.

## Features
- **Service Locator Global**: Central access to services.
- **BaseSystem Architecture**: Standardized startup/shutdown lifecycle.
- **Reactive Configuration**: Pydantic-based config with auto-save and events.
- **Async MongoDB ORM**: Advanced ODM with References, Embedding, and Indexing.
- **Capabilities System**: Driver/Plugin manager.
- **Digital Asset Management**: extensible asset ingestion.
- **Background Tasks**: Persistent task queue with history.
- **Journal**: Structured logging to database.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

## Structure
- `src/core`: Business logic and systems.
- `src/ui`: PySide6 widgets and windows.
- `src/core/database`: ORM and Manager.
- `src/core/tasks`: Background task system.

## Documentation
See `docs/` for detailed guides.
