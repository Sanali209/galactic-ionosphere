# Risks & Technical Debt

## Technical Debt

1.  **Synchronous Signal Usage**:
    -   **Context**: The legacy `Signal` / `ObserverEvent` class (in `src.core.events.observer`) is synchronous.
    -   **Impact**: Can block the UI if handlers perform heavy work.
    -   **Plan**: Migrate critical paths to async `EventBus`.

2.  **PyMongo Event Loop Compatibility**:
    -   **Context**: `motor` (PyMongo async) has issues with `qasync` event loops in some scenarios (specifically "no running event loop" during init).
    -   **Mitigation**: Initialize database connections lazily or strictly within the `qasync` loop context (fixed in recent sessions but fragile).

3.  **ORM Type Safety**:
    -   **Context**: The custom ORM (`src.core.database.orm`) lacks strict typing compared to full Pydantic V2 models.
    -   **Plan**: Evolve ORM to leverage Pydantic V2 fully.

## Risks

1.  **AI Performance on CPU**:
    -   **Risk**: Running CLIP/BLIP models on CPU blocks the application.
    -   **Mitigation**: Current implementation uses `ThreadPoolExecutor` but still competes for resources. Need a dedicated `ProcessPool` for true isolation.

2.  **Database Scalability**:
    -   **Risk**: MongoDB queries might slow down with >100k files without proper indexing.
    -   **Mitigation**: Ensure all queried fields (`tags`, `rating`, `date`) are indexed. Currently manual management.

3.  **Dependency Rot**:
    -   **Risk**: `facenet-pytorch` and other specific AI libs may become unmaintained.
    -   **Mitigation**: Abstract extractors behind interfaces (`Extractor` base class) to easily swap implementations.
