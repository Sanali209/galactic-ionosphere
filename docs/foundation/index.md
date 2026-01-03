# Foundation Overview

The Foundation (`src/core`) is the infrastructure layer of USCore. It provides the essential services required to build robust, scalable desktop applications.

## Core Components

| Component | Description | Doc Link |
| :--- | :--- | :--- |
| **Service Locator** | Dependency injection and lifecycle management. | [Service Locator](service_locator.md) |
| **Event System** | Decoupled communication via EventBus and Signals. | [Event System](event_system.md) |
| **Task System** | Persistent background task execution. | [Task System](task_system.md) |
| **Configuration** | Settings management via JSON. | [Configuration](configuration.md) |

## Directory Structure

-   **`bootstrap.py`**: Application entry point helper (`ApplicationBuilder`).
-   **`locator.py`**: Service Locator implementation.
-   **`base_system.py`**: Base class for all services.
-   **`events/`**: EventBus and Signal implementations.
-   **`tasks/`**: Background task processing.
-   **`database/`**: Database manager and ORM.
-   **`scheduling/`**: Periodic task scheduler (maintenance).
