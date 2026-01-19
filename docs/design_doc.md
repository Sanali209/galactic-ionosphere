[DESIGN_DOC]
Context:
- Problem: UI services submit tasks to a local `TaskSystem` instance which has no active workers. The Engine `TaskSystem` (with workers) never sees these tasks.
- Root Cause: `ApplicationBuilder` instantiates `TaskSystem` by default in the UI. `UCoreFSClientBundle` exclusion is bypassed by defaults.
- Constraints: Maintain `TaskSystem.submit` interface. Avoid complex architectural refactors of `DiscoveryService`.

Architecture:
- Components:
  - `TaskSystem` (UI): Acts as Client. Detects `EngineProxy`. Routes `submit()` calls to Engine.
  - `EngineProxy`: Bridge to Engine Loop.
  - `TaskSystem` (Engine): Acts as Server. Processes queue.
- Data flow:
  - UI Service -> `TaskSystem.submit()` -> (Check Proxy) -> `EngineProxy.submit()` -> Engine Loop -> `TaskSystem.submit()` -> DB + Queue.

Key Decisions:
- [D1] Lazy Proxy Detection in TaskSystem – `TaskSystem.submit` checks for `EngineProxy` in locator to determine if it should route remotely.
       Rationale: Simple, avoids configuration changes, handles the "One App, Two Threads" architecture seamlessly.

Interfaces:
- `TaskSystem.submit`: Unchanged signature. Behavior bifurcates based on environment (Client vs Engine).

Assumptions & TODOs:
- Assumptions: `EngineProxy` is only registered in the Main Thread. `EngineThread` does not have `EngineProxy`.
- TODOs (with priority):
  - [High] Modify `src/core/tasks/system.py` to implement routing logic in `submit`.
[/DESIGN_DOC]
