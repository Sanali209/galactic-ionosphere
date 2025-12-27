# Development Roadmap

This map outlines the development stages for the USCore project, aligning existing Linear tickets with strategic goals.

## Phase 1: Foundation Stability & Cleanup
*Goal: Ensure the codebase is production-ready, removing technical debt and obsolete patterns.*

-   [ ] **[SAN-31] Refactored: Remove Obsolete ChromaDB**
    -   *Context*: Remove obsolete references from `src/ucorefs/README.md`. Code is already clean.
    -   *Status*: Todo (High Priority)
-   [ ] **[SAN-32] Update Legacy Documentation Paths**
    -   *Context*: Fix strict path references in `docs/` to match `src/` structure.
    -   *Status*: Backlog (Created by Agent)
-   [ ] **[SAN-6] Integration Folder Migration**
    -   *Context*: Move stable code from `integration/` to `foundation/src/`.
    -   *Status*: Todo

## Phase 2: Architectural Standardization
*Goal: Enforce SOLID/DRY principles across the framework foundation.*

-   [ ] **[SAN-19] USCore Architectural Audit & Refactor**
    -   *Tasks*:
        -   **Events**: Deprecate `ObserverEvent` (sync) in favor of `EventBus` (unified).
        -   **ORM**: Fix inheritance gaps and enforce rigid `DatabaseManager` usage.
        -   **Logging**: Standardize logging/config.
    -   *Status*: Todo
-   [ ] **[SAN-14] Task Execution Optimization**
    -   *Context*: Offload CPU-heavy tasks to thread pool in `TaskSystem` to prevent UI blocking.
    -   *Status*: Backlog (High Priority)

## Phase 3: Feature Implementation
*Goal: Complete missing core features.*

-   [ ] **[SAN-23] DockingService Auto-Hide**
    -   *Context*: Implement "unpin" functionality for dock panels.
    -   *Status*: Todo
-   [ ] **[SAN-5] Docking Persistence Integration**
    -   *Context*: Verify docking persistence in UExplorer.
    -   *Status*: In Progress

## Phase 4: Future / Unassigned
-   **Unit Test Expansion**: (No ticket yet) - Need to increase coverage for `FSService` and `EventBus`.
-   **Antigravity/Physics Systems**: (Deferred per user request).
