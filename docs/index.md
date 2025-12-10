# Architecture Overview

The `Local Gallery & AI Manager` is built on a modular "Foundation Layer" designed for clean separation of concerns, reactivity, and independence from the UI framework (Qt).

## Core Principles

1.  **Clean Architecture**: The core logic (`src/core`) knows nothing about the UI (`src/ui`). Communication happens via a `ServiceLocator` and a global Event Bus.
2.  **Reactivity**: Settings and Data Models emit events (`on_change`) when modified, allowing the UI to update automatically.
3.  **Modularity**: Capabilities (AI, Storage) are implemented as swappable "Drivers".

## Directory Structure

-   `src/core/events.py`: **Event System** (Observer Pattern).
-   `src/core/messaging/`: **Messaging Protocol** (SystemMessage).
-   `src/core/config.py`: **Reactive Configuration** (Pydantic).
-   `src/core/capabilities/`: **Driver Facade**.
-   `src/core/database/`: **Mongo ORM** (Async/Motor).
-   `src/core/locator.py`: **Service Locator** (Singleton).

## Quick Start

Initialize the system:

```python
from src.core.locator import sl

# Initialize Core
sl.init("settings.json")

# Connect to Database
sl.caps.db.init()
```
