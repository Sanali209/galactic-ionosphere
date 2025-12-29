# Maintenance Tasks & Task System Research - Session Journal

**Date**: 2025-12-28  
**Session ID**: maintenance_research  
**Status**: Complete

## [PROGRESS]
- Phase: Execution
- Step: Documentation complete
- Completed: 13/13 steps
- Next: Review with user
[/PROGRESS]

## Objective

Deep research on maintenance tasks for UCore FS and UExplorer, including:
- What maintenance tasks are currently implemented
- What tasks need to be or could be implemented
- How to run them periodically
- Whether to change the task system

## Research Findings

### 1. Currently Implemented Maintenance Tasks

#### MaintenanceService Operations

**Location**: `src/ucorefs/services/maintenance_service.py`

The system has a centralized `MaintenanceService` with **5 core maintenance operations**:

1. **`rebuild_all_counts()`** - Recalculate file counts across Tag, Album, and Directory systems
   - Purpose: Fix count drift after bulk operations or database migrations
   - Returns: Tags updated, albums updated, directories updated, duration, errors
   - **Status**: ✅ Implemented

2. **`rebuild_album_references()`** - Rebuild FileRecord.album_ids from Album.file_ids
   - Purpose: Sync bidirectional relationship between albums and files
   - Use case: Run after upgrading to bidirectional album tracking
   - **Status**: ✅ Implemented

3. **`verify_references()`** - Verify all ObjectId references are valid
   - Checks: FileRecord.tag_ids, FileRecord.album_ids, FileRecord.parent_id
   - Returns: Counts of broken references
   - **Status**: ✅ Implemented

4. **`cleanup_orphaned_records()`** - Remove references to deleted records
   - Cleans: FileRecord.tag_ids, FileRecord.album_ids, Album.file_ids
   - Returns: Files cleaned, references removed
   - **Status**: ✅ Implemented

5. **`background_count_verification()`** - Background task for periodic count verification
   - Purpose: Silently fix count drift during app idle time
   - Design: To be called every 5 minutes of idle time
   - **Status**: ⚠️ **Implemented but NOT SCHEDULED** (no scheduler exists)

### 2. Task System Architecture

#### Current TaskSystem Implementation

**Location**: `src/core/tasks/system.py`

**Key Features**:
- **Priority Queue**: Uses `asyncio.PriorityQueue` (SAN-14 Phase 3)
- **Priority Levels**: 0=HIGH, 1=NORMAL (default), 2=LOW
- **Worker Count**: 8 workers (increased from 3 for better concurrency)
- **Crash Recovery**: Tasks persisted to MongoDB, auto-rescheduled on startup
- **Exponential Backoff**: 1s → 2s → 4s → 8s → 16s (max 30s)
- **Thread Pool Offloading**: Sync handlers automatically offloaded to thread pool

**Current Limitations**:
- ❌ **No periodic/scheduled task execution**
- ❌ **No cron-like scheduling**
- ❌ **No idle detection mechanism**
- ❌ **No recurring task support**
- Tasks are only submitted on-demand via `submit()` method

### 3. Missing Maintenance Tasks

#### Based on Documentation Analysis

**High Priority Missing Tasks** (from `docs/maintenance_tasks_ucorefs_uexplorer.md`):

1. **Viewport Priority Queue UI Integration** ⏱️ 2-3 hours
   - Status: Backend complete (SAN-14), UI integration deferred
   - Impact: Significantly improves perceived performance

2. **Database Maintenance** (partially implemented)
   - Missing: Remove obsolete file records for deleted files
   - Missing: Vacuum/optimize database
   - Missing: FAISS index optimization/rebuild

3. **Cache Management** (not implemented)
   - Missing: Thumbnail cache cleanup
   - Missing: Temporary file cleanup
   - Missing: Log file rotation/cleanup

4. **Index Maintenance** (not implemented)
   - Missing: MongoDB index analysis and optimization
   - Missing: FAISS vector index rebuild
   - Missing: BK-Tree rebuild (if persisted)

5. **AI Model Cache Management** (not implemented)
   - Missing: Clear model cache
   - Missing: Verify model checksums
   - Missing: Re-download corrupted models

#### Additional Potential Tasks

6. **Health Checks** (not implemented)
   - Database connectivity check
   - FAISS index health check
   - File system accessibility check
   - GPU availability check

7. **Performance Metrics Collection** (not implemented)
   - Task execution statistics
   - Database query performance
   - Memory usage tracking
   - Processing throughput metrics

8. **Backup/Export** (not implemented)
   - Database backup
   - Configuration backup
   - Export metadata to JSON/CSV

### 4. Periodic Execution Options

#### Option 1: Simple Asyncio Loop (Recommended for Initial Implementation)

**Pros**:
- ✅ No external dependencies
- ✅ Simple to implement
- ✅ Integrated with existing `TaskSystem`

**Cons**:
- ⚠️ Not persistent (stops when app stops)
- ⚠️ Manual scheduling logic

**Implementation**:
```python
class PeriodicTaskScheduler(BaseSystem):
    async def initialize(self):
        # Schedule periodic tasks
        asyncio.create_task(self._run_periodic_maintenance())
    
    async def _run_periodic_maintenance(self):
        while self._running:
            # Run every 5 minutes (300 seconds)
            await asyncio.sleep(300)
            
            maintenance = self.locator.get_system(MaintenanceService)
            await maintenance.background_count_verification()
```

#### Option 2: APScheduler (Full-Featured Scheduler)

**External Dependency**: `pip install apscheduler`

**Pros**:
- ✅ Cron-like scheduling
- ✅ Job persistence (can survive app restart)
- ✅ Multiple triggers (interval, cron, date)
- ✅ Timezone support

**Cons**:
- ❌ External dependency
- ❌ More complex setup

**Implementation**:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

class PeriodicTaskScheduler(BaseSystem):
    async def initialize(self):
        self._scheduler = AsyncIOScheduler()
        
        # Every 5 minutes
        self._scheduler.add_job(
            lambda: self._run_background_verification(),
            trigger=IntervalTrigger(minutes=5),
            id='background_count_verification'
        )
        
        # Every day at 3 AM (database optimization)
        self._scheduler.add_job(
            lambda: self._run_database_optimization(),
            trigger='cron',
            hour=3,
            minute=0,
            id='database_optimization'
        )
        
        self._scheduler.start()
```

#### Option 3: Idle Detection (UI-Based)

**Use Case**: Run maintenance when user is idle

**Implementation**:
```python
# In main UI window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._idle_timer = QTimer()
        self._idle_timer.timeout.connect(self._on_idle)
        self._last_activity = time.time()
    
    def eventFilter(self, obj, event):
        # Reset idle timer on any user activity
        if event.type() in (QEvent.MouseMove, QEvent.KeyPress):
            self._last_activity = time.time()
        return super().eventFilter(obj, event)
    
    def _on_idle(self):
        # Check if 5 minutes of idle
        if time.time() - self._last_activity > 300:
            maintenance = self.locator.get_system(MaintenanceService)
            asyncio.ensure_future(maintenance.background_count_verification())
```

#### Option 4: Hybrid Approach (Recommended)

Combine simple asyncio loop with configurable scheduling:

```python
class PeriodicTaskScheduler(BaseSystem):
    """Lightweight periodic task scheduler without external dependencies."""
    
    async def initialize(self):
        # Read schedule from config
        self._schedule = self.config.get('maintenance.schedule', {
            'background_verification': {'interval_minutes': 5},
            'database_optimization': {'interval_hours': 24},
            'cache_cleanup': {'interval_hours': 6}
        })
        
        # Start schedulers
        for task_name, schedule in self._schedule.items():
            asyncio.create_task(self._run_periodic_task(task_name, schedule))
    
    async def _run_periodic_task(self, task_name, schedule):
        interval_seconds = (
            schedule.get('interval_minutes', 0) * 60 +
            schedule.get('interval_hours', 0) * 3600
        )
        
        while self._running:
            await asyncio.sleep(interval_seconds)
            
            # Execute task via TaskSystem
            await self.task_system.submit(
                f'maintenance_{task_name}',
                f'Maintenance: {task_name}',
                priority=TaskSystem.PRIORITY_LOW
            )
```

### 5. Should the Task System Be Changed?

#### Current Assessment: **NO MAJOR CHANGES NEEDED** ✅

The current `TaskSystem` is well-designed and production-ready after SAN-14 optimizations:

**Strengths**:
- ✅ Priority queue implemented (SAN-14 Phase 3)
- ✅ Crash recovery via MongoDB persistence
- ✅ Thread pool offloading for CPU-heavy tasks
- ✅ Configurable worker count (8 workers)
- ✅ Exponential backoff for resilience
- ✅ Progress tracking infrastructure

**Only Missing Feature**: **Periodic/Scheduled Execution**

#### Recommended Enhancement: Add Scheduling Layer

Instead of changing `TaskSystem`, **add a lightweight scheduling layer on top**:

```
┌─────────────────────────────────┐
│  PeriodicTaskScheduler          │  ← NEW (lightweight)
│  - Asyncio loops                │
│  - Configurable intervals       │
│  - Submits to TaskSystem        │
└──────────┬──────────────────────┘
           │ submit(handler, ...)
           ↓
┌─────────────────────────────────┐
│  TaskSystem                     │  ← EXISTING (no changes)
│  - Priority queue               │
│  - Crash recovery               │
│  - Worker pool                  │
└─────────────────────────────────┘
```

**Benefits**:
- ✅ Preserves existing TaskSystem design
- ✅ Scheduling is optional (not all apps need it)
- ✅ Easy to disable/configure
- ✅ No breaking changes

## Recommendations

### Short-Term (1-2 weeks)

1. **Implement `PeriodicTaskScheduler`** (Option 4 - Hybrid Approach)
   - No external dependencies
   - Configurable via `config.json`
   - Starts with 2-3 essential tasks:
     - Background count verification (every 5 min)
     - Database optimization (every 24 hours)
     - Cache cleanup (every 6 hours)

2. **Add Missing Maintenance Tasks**:
   - Database vacuum/optimize
   - Thumbnail cache cleanup
   - Log file rotation
   - Orphaned file record cleanup

3. **Create Maintenance Dashboard** (UExplorer UI)
   - Show last run time for each task
   - Manual trigger buttons
   - Task execution history
   - Current maintenance status

### Medium-Term (1-2 months)

4. **Add Health Checks**
   - Database connectivity
   - FAISS index integrity
   - File system accessibility
   - GPU availability

5. **Implement Configuration Auto-Tuning** (from Roadmap Phase 5)
   - Detect system capabilities
   - Auto-adjust batch sizes
   - Performance profiles

6. **Add Metrics Collection**
   - Task execution statistics
   - Performance tracking
   - Resource usage monitoring

### Long-Term (3-6 months)

7. **Consider APScheduler** (if needed)
   - Only if cron-like scheduling becomes critical
   - Only if job persistence across restarts is needed
   - Evaluate trade-off of external dependency

8. **Implement Advanced Scheduling**
   - Time-of-day preferences (run heavy tasks at night)
   - Resource-aware scheduling (don't run if GPU busy)
   - User activity detection (run maintenance when idle)

## Configuration Design

### Proposed config.json Structure

```json
{
  "maintenance": {
    "enabled": true,
    "schedule": {
      "background_verification": {
        "enabled": true,
        "interval_minutes": 5
      },
      "database_optimization": {
        "enabled": true,
        "interval_hours": 24
      },
      "cache_cleanup": {
        "enabled": true,
        "interval_hours": 6,
        "max_cache_size_gb": 10
      },
      "thumbnail_cleanup": {
        "enabled": true,
        "interval_hours": 12,
        "max_age_days": 30
      },
      "log_rotation": {
        "enabled": true,
        "interval_hours": 24,
        "max_log_files": 10,
        "max_log_size_mb": 100
      }
    }
  }
}
```

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **High** | Implement PeriodicTaskScheduler | 4-6 hours | High (enables all periodic tasks) |
| **High** | Add database optimization task | 2-3 hours | High (performance) |
| **High** | Add cache cleanup tasks | 2-3 hours | Medium (disk space) |
| **Medium** | Create Maintenance Dashboard UI | 4-6 hours | Medium (visibility) |
| **Medium** | Add health checks | 3-4 hours | Medium (reliability) |
| **Low** | Add metrics collection | 1-2 days | Low (analytics) |

## References

- [MaintenanceService](file:///d:/github/USCore/src/ucorefs/services/maintenance_service.py)
- [TaskSystem](file:///d:/github/USCore/src/core/tasks/system.py)
- [Existing Maintenance Tasks Doc](file:///d:/github/USCore/docs/maintenance_tasks_ucorefs_uexplorer.md)
- [TaskSystem SAN-14 Optimization](file:///d:/github/USCore/docs/tasksystem_san14_optimization.md)
- [Roadmap](file:///d:/github/USCore/docs/roadmap.md)

---

**Session End Time**: 2025-12-28 13:15:00  
**Duration**: ~15 minutes  
**Output**: Comprehensive maintenance tasks research and periodic scheduler recommendations
