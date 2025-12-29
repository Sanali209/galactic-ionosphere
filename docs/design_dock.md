# Design Document - USCore Foundation

[DESIGN_DOC]
Context:
- Problem: Complex project analysis and lack of centralized design documentation.
- Constraints: Follow USCore conventions.
- Non-goals: Complete refactoring (this session is for analysis).

Architecture:
- Components:
  - **Core Foundation**:
    - `ServiceLocator`: Handles dependency injection and lifecycle (topological sort startup).
    - `EventBus`: Centralized event system for cross-module communication.
    - `CommandBus`: Executes decoupled commands.
    - `TaskSystem`: Async background task management using `qasync`.
  - **UI Layer (Qt/MVVM)**:
    - `DockingService`: Advanced docking using `pyqtdocking`.
    - `MVVM`: `BindableProperty` and `ViewModelProvider` for reactive UI.
    - `CardView`: Specialized widget for displaying data items with thumbnails.
  - **UCoreFS (Intelligence Layer)**:
    - File database with watchers and crawlers.
    - AI extractors (CLIP, BLIP, YOLOv8) for metadata enrichment.
    - Vector search (ChromaDB) for semantic retrieval.
  - **NodeGraph (Visual Engine)**:
    - Flow control and execution of visual scripting nodes.

- Data flow:
  - `OS` -> `UCoreFS` (Crawler/Watchdog) -> `Extractors` -> `DB` mongo db croma is obsoleted
  - `UI` -> `ViewModel` -> `QueryEngine` -> `DB` -> `UI` (MVVM Update)

- External dependencies:
  - PySide6, qasync, loguru.
  - pyqtdocking (QtAds).
  - OpenCV, PyTorch (for AI extractors).
  - ChromaDB (vector database).

Key Decisions:
- [D1] Initialize dev_log and design_dock for better traceability.
- [D2] Move from singleton services to ServiceLocator-managed systems for testability.
- [D3] Use topological sorting for system startup to ensure dependencies are ready.

Interfaces:
- `BaseSystem`: Lifecycle interface (`initialize`, `shutdown`).
- `Command`: Interface for bus-executed actions.
- `Event`: Data structure for bus broadcasts.

Assumption & Constraints:
- **Priorities from Linear**:
  - `SAN-31`: Refactor/remove obsolete ChromaDB references (High Priority Tech Debt).
  - `SAN-19`: Architectural Audit & Refactor (Ongoing).
  - `SAN-29`, `SAN-28`: UI/Docking configuration (Recently Done).
  - `SAN-18`, `SAN-17`: Documentation backfill (Recently Done).

Assumptions & TODOs:
- Assumptions: The project is being refactored for enterprise standards (SAN-19).
- Open questions: Best migration strategy for removing ChromaDB?
- TODOs (with priority):
  - [High] Implement PeriodicTaskScheduler for automated maintenance (2025-12-28)
    - Research complete: See [maintenance_periodic_execution.md](maintenance_periodic_execution.md)
    - Decision: Simple asyncio loop (no external deps), configurable intervals
    - Tasks: background_verification (5min), database_optimization (24hr), cache_cleanup (6hr)
  - [High] Complete analysis of `src/` subdirectories.
  - [Med] Map dependencies between modules.
  - [Med] Add missing maintenance tasks (database optimization, cache cleanup, log rotation)
  - [Low] Update out-of-date documentation.
  - [Linear] Address `SAN-31`: Remove obsolete ChromaDB references.
[/DESIGN_DOC]
