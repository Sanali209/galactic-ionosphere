# API Reference

Direct links to source code modules.

## Core
- **Base System**: [`base_system.py`](../src/core/base_system.py)
- **Locator**: [`locator.py`](../src/core/locator.py)
- **Configuration**: [`config.py`](../src/core/config.py)
- **Events**: [`events.py`](../src/core/events.py)

## Subsystems
- **Database**: 
    - [`manager.py`](../src/core/database/manager.py)
    - [`orm.py`](../src/core/database/orm.py)
- **Tasks**: 
    - [`system.py`](../src/core/tasks/system.py)
    - [`models.py`](../src/core/tasks/models.py)
- **Assets**: 
    - [`manager.py`](../src/core/assets/manager.py)
    - [`base.py`](../src/core/assets/base.py)
- **Commands**: 
    - [`bus.py`](../src/core/commands/bus.py)
    - [`base.py`](../src/core/commands/base.py)
- **Journal**: 
    - [`service.py`](../src/core/journal/service.py)
    - [`models.py`](../src/core/journal/models.py)

## UI
- [`main_window.py`](../src/ui/main_window.py)
- [`bridge.py`](../src/ui/bridge.py)
