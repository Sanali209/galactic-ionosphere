# Maintenance Tasks: Periodic Execution Guide

**Last Updated**: 2025-12-28  
**Status**: Proposed Design

## Overview

This document provides analysis and recommendations for implementing periodic execution of maintenance tasks in UCore FS and UExplorer.

## Current State

### ✅ What's Implemented

1. **MaintenanceService** with 5 core operations:
   - `rebuild_all_counts()` - Fix count drift
   - `rebuild_album_references()` - Sync album relationships
   - `verify_references()` - Check ObjectId integrity
   - `cleanup_orphaned_records()` - Remove broken references
   - `background_count_verification()` - Periodic verification (⚠️ **not scheduled**)

2. **TaskSystem** (production-ready after SAN-14):
   - Priority queue (HIGH/NORMAL/LOW)
   - 8 async workers
   - Crash recovery via MongoDB
   - Thread pool offloading
   - Exponential backoff

### ❌ What's Missing

- **No periodic/scheduled task execution mechanism**
- No cron-like scheduling
- No idle detection
- No recurring task support

> **Note**: `background_count_verification()` is designed to run "every 5 minutes of idle time" but there's no scheduler to actually run it.

---

## Periodic Execution Options

### Option 1: Simple Asyncio Loop ⭐ **Recommended for Initial Implementation**

**Implementation**:

```python
# src/core/scheduling/periodic_scheduler.py
from typing import Dict, Any
import asyncio
from loguru import logger
from src.core.base_system import BaseSystem

class PeriodicTaskScheduler(BaseSystem):
    """Lightweight periodic task scheduler without external dependencies."""
    
    depends_on = []  # Optional: add TaskSystem if submitting via TaskSystem
    
    def __init__(self, locator, config):
        super().__init__(locator, config)
        self._running = False
        self._scheduled_tasks = []
    
    async def initialize(self):
        logger.info("PeriodicTaskScheduler initializing...")
        self._running = True
        
        # Read schedule from config
        schedule = self._get_schedule_config()
        
        # Start each scheduled task
        for task_name, config in schedule.items():
            if config.get('enabled', True):
                task_coroutine = asyncio.create_task(
                    self._run_periodic_task(task_name, config)
                )
                self._scheduled_tasks.append(task_coroutine)
                logger.info(f"Scheduled periodic task: {task_name}")
        
        await super().initialize()
    
    async def shutdown(self):
        logger.info("PeriodicTaskScheduler shutting down...")
        self._running = False
        
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        await super().shutdown()
    
    def _get_schedule_config(self) -> Dict[str, Any]:
        """Read maintenance schedule from config."""
        default_schedule = {
            'background_verification': {
                'enabled': True,
                'interval_minutes': 5,
                'handler': 'maintenance_background_verification'
            },
            'database_optimization': {
                'enabled': True,
                'interval_hours': 24,
                'handler': 'maintenance_database_optimization'
            },
            'cache_cleanup': {
                'enabled': True,
                'interval_hours': 6,
                'handler': 'maintenance_cache_cleanup'
            }
        }
        
        if hasattr(self.config.data, 'maintenance'):
            return getattr(self.config.data.maintenance, 'schedule', default_schedule)
        
        return default_schedule
    
    async def _run_periodic_task(self, task_name: str, config: Dict[str, Any]):
        """Run a periodic task on configured interval."""
        # Calculate interval in seconds
        interval_seconds = (
            config.get('interval_minutes', 0) * 60 +
            config.get('interval_hours', 0) * 3600 +
            config.get('interval_days', 0) * 86400
        )
        
        if interval_seconds == 0:
            logger.warning(f"Task {task_name} has zero interval, skipping")
            return
        
        # Initial delay (optional, to stagger task starts)
        initial_delay = config.get('initial_delay_seconds', 0)
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        
        logger.info(f"Starting periodic task: {task_name} (interval: {interval_seconds}s)")
        
        while self._running:
            try:
                # Execute task
                await self._execute_task(task_name, config)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"Periodic task cancelled: {task_name}")
                break
            except Exception as e:
                logger.error(f"Error in periodic task {task_name}: {e}")
                # Continue running even if task fails
                await asyncio.sleep(interval_seconds)
    
    async def _execute_task(self, task_name: str, config: Dict[str, Any]):
        """Execute a single maintenance task."""
        handler_name = config.get('handler')
        
        if handler_name == 'maintenance_background_verification':
            maintenance = self.locator.get_system('MaintenanceService')
            if maintenance:
                await maintenance.background_count_verification()
        
        elif handler_name == 'maintenance_database_optimization':
            # TODO: Implement database optimization
            logger.info("Database optimization task (not yet implemented)")
        
        elif handler_name == 'maintenance_cache_cleanup':
            # TODO: Implement cache cleanup
            logger.info("Cache cleanup task (not yet implemented)")
        
        else:
            logger.warning(f"Unknown maintenance handler: {handler_name}")
```

**Pros**:
- ✅ No external dependencies
- ✅ Simple to implement and maintain
- ✅ Integrated with existing architecture
- ✅ Configurable via `config.json`

**Cons**:
- ⚠️ Not persistent (stops when app stops)
- ⚠️ Manual configuration (no cron syntax)

---

### Option 2: APScheduler (Full-Featured)

**External Dependency**: `pip install apscheduler`

**Implementation**:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

class PeriodicTaskScheduler(BaseSystem):
    async def initialize(self):
        self._scheduler = AsyncIOScheduler()
        
        # Register maintenance tasks
        self._register_maintenance_tasks()
        
        # Start scheduler
        self._scheduler.start()
        await super().initialize()
    
    def _register_maintenance_tasks(self):
        # Every 5 minutes
        self._scheduler.add_job(
            self._run_background_verification,
            trigger=IntervalTrigger(minutes=5),
            id='background_count_verification'
        )
        
        # Every day at 3 AM (database optimization)
        self._scheduler.add_job(
            self._run_database_optimization,
            trigger=CronTrigger(hour=3, minute=0),
            id='database_optimization'
        )
        
        # Every 6 hours (cache cleanup)
        self._scheduler.add_job(
            self._run_cache_cleanup,
            trigger=IntervalTrigger(hours=6),
            id='cache_cleanup'
        )
    
    async def _run_background_verification(self):
        maintenance = self.locator.get_system('MaintenanceService')
        await maintenance.background_count_verification()
```

**Pros**:
- ✅ Cron-like scheduling
- ✅ Job persistence (can survive app restart with JobStore)
- ✅ Multiple triggers (interval, cron, date)
- ✅ Timezone support
- ✅ Battle-tested library

**Cons**:
- ❌ External dependency
- ❌ More complex setup
- ❌ Overkill for simple use cases

---

### Option 3: Idle Detection (UI-Based)

**Use Case**: Run maintenance only when user is idle

**Implementation**:

```python
# In UExplorer main window
from PySide6.QtCore import QTimer, QEvent
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Idle timer
        self._idle_timer = QTimer()
        self._idle_timer.setInterval(60000)  # Check every minute
        self._idle_timer.timeout.connect(self._check_idle)
        self._idle_timer.start()
        
        self._last_activity = time.time()
        self._idle_threshold = 300  # 5 minutes
        
        # Install event filter to track activity
        QApplication.instance().installEventFilter(self)
    
    def eventFilter(self, obj, event):
        # Reset idle timer on any user activity
        if event.type() in (QEvent.MouseMove, QEvent.KeyPress, QEvent.MouseButtonPress):
            self._last_activity = time.time()
        return super().eventFilter(obj, event)
    
    def _check_idle(self):
        idle_time = time.time() - self._last_activity
        
        if idle_time > self._idle_threshold:
            # User is idle, run maintenance
            self._run_idle_maintenance()
            # Reset to avoid running multiple times
            self._last_activity = time.time()
    
    def _run_idle_maintenance(self):
        maintenance = self.locator.get_system('MaintenanceService')
        asyncio.ensure_future(maintenance.background_count_verification())
```

**Pros**:
- ✅ Doesn't interrupt user work
- ✅ No external dependencies

**Cons**:
- ⚠️ Only runs when UI is open
- ⚠️ Doesn't run in background/service mode
- ⚠️ Complex event filtering

---

### ⭐ Recommended: Hybrid Approach

Combine **Option 1 (Simple Asyncio)** with **Option 3 (Idle Detection)**:

1. **PeriodicTaskScheduler** (asyncio-based) for critical tasks that must run
2. **Idle detection** for optional/heavy tasks that should wait for user inactivity

**Configuration**:

```json
{
  "maintenance": {
    "enabled": true,
    "use_idle_detection": true,
    "idle_threshold_minutes": 5,
    "schedule": {
      "background_verification": {
        "enabled": true,
        "interval_minutes": 5,
        "run_on_idle_only": false
      },
      "database_optimization": {
        "enabled": true,
        "interval_hours": 24,
        "run_on_idle_only": true,
        "preferred_hour": 3
      },
      "cache_cleanup": {
        "enabled": true,
        "interval_hours": 6,
        "run_on_idle_only": false
      }
    }
  }
}
```

---

## Proposed New Maintenance Tasks

### 1. Database Optimization

```python
async def database_optimization(self):
    """Optimize MongoDB and FAISS indexes."""
    from src.core.database.manager import DatabaseManager
    
    db_manager = self.locator.get_system(DatabaseManager)
    
    # MongoDB: Compact collections
    for collection_name in ['files', 'tags', 'albums', 'directories']:
        await db_manager.db[collection_name].compact()
    
    # MongoDB: Rebuild indexes
    await db_manager.ensure_indexes()
    
    # FAISS: Rebuild index if needed
    # TODO: Add FAISS optimization
    
    logger.info("Database optimization complete")
```

### 2. Cache Cleanup

```python
async def cache_cleanup(self, max_cache_size_gb=10, max_age_days=30):
    """Clean up old thumbnails and temporary files."""
    import shutil
    from pathlib import Path
    
    # Thumbnail cache
    thumbnail_dir = Path("./thumbnails")
    current_size = sum(f.stat().st_size for f in thumbnail_dir.rglob('*') if f.is_file())
    current_size_gb = current_size / (1024**3)
    
    if current_size_gb > max_cache_size_gb:
        # Delete oldest files
        files = sorted(thumbnail_dir.rglob('*'), key=lambda f: f.stat().st_mtime)
        for file in files:
            if current_size_gb <= max_cache_size_gb:
                break
            file.unlink()
            current_size_gb -= file.stat().st_size / (1024**3)
    
    # Delete old thumbnails
    cutoff_time = time.time() - (max_age_days * 86400)
    for file in thumbnail_dir.rglob('*'):
        if file.stat().st_mtime < cutoff_time:
            file.unlink()
    
    logger.info(f"Cache cleanup complete. Size: {current_size_gb:.2f}GB")
```

### 3. Log Rotation

```python
async def log_rotation(self, max_log_files=10, max_log_size_mb=100):
    """Rotate log files."""
    from pathlib import Path
    
    log_dir = Path("./logs")
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    # Delete old logs
    for i, log_file in enumerate(log_files):
        if i >= max_log_files:
            log_file.unlink()
        elif log_file.stat().st_size > max_log_size_mb * 1024 * 1024:
            # Archive large log
            archive_path = log_file.with_suffix('.log.old')
            log_file.rename(archive_path)
    
    logger.info(f"Log rotation complete. Kept {min(len(log_files), max_log_files)} files")
```

### 4. Orphaned File Records Cleanup

```python
async def cleanup_orphaned_file_records(self):
    """Remove FileRecords for files that no longer exist on disk."""
    from src.ucorefs.models.file_record import FileRecord
    from pathlib import Path
    
    removed_count = 0
    all_files = await FileRecord.find({})
    
    for file_record in all_files:
        if not Path(file_record.path).exists():
            await file_record.delete()
            removed_count += 1
    
    logger.info(f"Removed {removed_count} orphaned file records")
    return removed_count
```

---

## Integration Plan

### Step 1: Create PeriodicTaskScheduler

1. Create `src/core/scheduling/periodic_scheduler.py` (Option 1 implementation)
2. Register in application bootstrap (UExplorer `main.py`)
3. Add configuration section to `config.json`

### Step 2: Register Maintenance Task Handlers

Update `MaintenanceService` to register handlers with `TaskSystem`:

```python
async def initialize(self):
    await super().initialize()
    
    # Register handlers with TaskSystem
    task_system = self.locator.get_system('TaskSystem')
    if task_system:
        task_system.register_handler('maintenance_background_verification', 
                                     self.background_count_verification)
        task_system.register_handler('maintenance_database_optimization', 
                                     self.database_optimization)
        task_system.register_handler('maintenance_cache_cleanup', 
                                     self.cache_cleanup)
        task_system.register_handler('maintenance_orphaned_cleanup', 
                                     self.cleanup_orphaned_file_records)
```

### Step 3: Add UI Dashboard (Optional)

Create `MaintenancePanel` in UExplorer to show:
- Last run time for each task
- Manual trigger buttons
- Task execution history
- Enable/disable toggles

---

## Configuration Schema

### config.json

```json
{
  "maintenance": {
    "enabled": true,
    "use_idle_detection": true,
    "idle_threshold_minutes": 5,
    "schedule": {
      "background_verification": {
        "enabled": true,
        "interval_minutes": 5,
        "handler": "maintenance_background_verification",
        "run_on_idle_only": false
      },
      "database_optimization": {
        "enabled": true,
        "interval_hours": 24,
        "handler": "maintenance_database_optimization",
        "run_on_idle_only": true,
        "preferred_hour": 3,
        "initial_delay_seconds": 300
      },
      "cache_cleanup": {
        "enabled": true,
        "interval_hours": 6,
        "handler": "maintenance_cache_cleanup",
        "run_on_idle_only": false,
        "max_cache_size_gb": 10,
        "max_age_days": 30
      },
      "orphaned_cleanup": {
        "enabled": true,
        "interval_hours": 12,
        "handler": "maintenance_orphaned_cleanup",
        "run_on_idle_only": true
      },
      "log_rotation": {
        "enabled": true,
        "interval_hours": 24,
        "handler": "maintenance_log_rotation",
        "max_log_files": 10,
        "max_log_size_mb": 100
      }
    }
  }
}
```

---

## Summary

### ✅ Recommended Approach

1. **Implement PeriodicTaskScheduler** using Option 1 (Simple Asyncio Loop)
2. **Add 4-5 new maintenance tasks** (database optimization, cache cleanup, etc.)
3. **Integrate with existing TaskSystem** (submit tasks with LOW priority)
4. **Make everything configurable** via `config.json`
5. **Optional**: Add idle detection for heavy tasks
6. **Optional**: Add UI dashboard for visibility

### 📊 Effort Estimate

| Task | Effort |
|------|--------|
| Implement PeriodicTaskScheduler | 4-6 hours |
| Add database optimization | 2-3 hours |
| Add cache cleanup tasks | 2-3 hours |
| Add orphaned record cleanup | 1-2 hours |
| Add log rotation | 1-2 hours |
| Configuration schema | 1 hour |
| Testing | 2-3 hours |
| **Total** | **13-20 hours** |

### 🚀 Benefits

- ✅ Automated maintenance (no manual intervention)
- ✅ Better performance (regular optimization)
- ✅ Disk space management (cache/log cleanup)
- ✅ Data integrity (orphaned record cleanup)
- ✅ Configurable (users can enable/disable tasks)
- ✅ No external dependencies (pure Python + asyncio)

---

## References

- [MaintenanceService](file:///d:/github/USCore/src/ucorefs/services/maintenance_service.py)
- [TaskSystem](file:///d:/github/USCore/src/core/tasks/system.py)
- [Session Journal](file:///d:/github/USCore/dev_log/journal_session_maintenance_research.md)
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [APScheduler docs](https://apscheduler.readthedocs.io/) (if needed later)
