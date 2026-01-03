# System Architecture Overview

## High-Level Design

USCore is a modular application framework built on Python and Qt (PySide6). It follows a layered architecture designed for scalability, testability, and separation of concerns.

```mermaid
graph TD
    User[User] --> UI[UI Layer (Qt/MVVM)]
    UI --> App[Application Layer]
    App --> Foundation[Foundation Core]
    
    subgraph Foundation
        Foundation --> SL[Service Locator]
        Foundation --> Bus[Event/Command Bus]
        Foundation --> Tasks[Task System]
    end
    
    subgraph Modules
        Foundation --> UCoreFS[UCoreFS (Data & AI)]
        Foundation --> NodeGraph[NodeGraph (Visual Scripting)]
    end
    
    UCoreFS --> DB[(MongoDB)]
    UCoreFS --> VectorDB[(FAISS)]
```

### Layers

1.  **UI Layer**:
    -   Built with **PySide6**.
    -   Uses **MVVM** pattern (`BindableProperty`, `ViewModelProvider`).
    -   Features a professional **Docking System** (`PySide6-QtAds`).

2.  **Application Layer**:
    -   **Bootstrapping**: `ApplicationBuilder` configures and composes the app.
    -   **SystemBundles**: Groups related services (e.g., `UCoreFSBundle`) for easy registration.

3.  **Foundation Core** (`src/core`):
    -   The kernel of the framework.
    -   **Service Locator**: Handles dependency injection and lifecycle management.
    -   **BaseSystem**: Interface for all services with `initialize()` and `shutdown()` hooks.
    -   **Event System**: Async `EventBus` for decoupling and Sync `Signal` for tight loops.
    -   **Task System**: Async background processing with persistence and crash recovery.

4.  **Modules**:
    -   **UCoreFS**: A semantic filesystem layer. Indexes files, extracts metadata, manages tags/albums, and provides AI-powered search.
    -   **NodeGraph**: A visual programming engine (like Unreal Blueprints) for custom workflows.

## Key Design Patterns

-   **Dependency Injection**: All systems are registered in the `ServiceLocator` and request dependencies via `depends_on`.
-   **Topological Startup**: Services start in dependency order (Database -> EventBus -> ... -> UI).
-   **Event-Driven**: Systems communicate via the `EventBus` (`file.created`, `task.completed`), minimizing coupling.
-   **Async First**: The core is built on `asyncio` and `qasync` to keep the UI responsive during heavy I/O or AI tasks.
