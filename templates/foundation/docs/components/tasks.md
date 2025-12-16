# Task System

The Task System provides persistent, background task execution with crash recovery.

## Features
- **Async Execution**: Non-blocking task processing.
- **Persistence**: Tasks are stored in MongoDB (`tasks` collection).
- **Crash Recovery**: Interrupted tasks are automatically reset on restart.
- **History**: Full log of task status, result, and errors.

## Registration

To ensure tasks can be re-run after a restart (when memory is cleared), handlers must be registered by a unique name.

```python
from src.core.tasks.system import TaskSystem

async def resize_image(path: str, size: str):
    # ... logic ...
    return f"Resized {path}"

# Usage
task_sys = sl.get_system(TaskSystem)
task_sys.register_handler("resize_image", resize_image)
```

## Submitting Tasks

```python
# Arguments must be simple strings (for now) to ensure serialization
task_id = await task_sys.submit(
    "resize_image",   # Handler Name
    "Resizing Avatar", # Human-readable label
    "/path/to/img.jpg", "1024x768" # *args
)
```

## Monitoring

Tasks can be monitored via the `TaskRecord` model.

```python
from src.core.tasks.models import TaskRecord

task = await TaskRecord.get(task_id)
print(task.status) # pending, running, completed, failed
print(task.result)
print(task.progress)
```
