# TaskSystem Research Summary

**Date**: 2025-12-28  
**Research Objective**: Deep analysis of TaskSystem architecture and its relationship to the indexer pipeline for SAN-14 optimization

## Documentation Created

1. **[Technical Documentation](../docs/tasksystem_san14_optimization.md)** (500+ lines)
   - Complete TaskSystem architecture
   - CPU-heavy task offloading mechanisms
   - Integration with indexer pipeline
   - Detailed SAN-14 optimization plan

2. **[Session Journal](./journal_session_tasksystem_research.md)**
   - Detailed research notes
   - Code analysis
   - Performance findings

## Key Discoveries

### Architecture Overview

```
TaskSystem Architecture:
├─ Worker Pool: 3 async coroutines (default, configurable to 8+)
├─ Task Queue: asyncio.Queue (in-memory, unbounded)
├─ Persistence: MongoDB TaskRecord for crash recovery
├─ Thread Pool: Automatic offloading for sync handlers
└─ Integration: DiscoveryService + ProcessingPipeline
```

### How CPU-Heavy Tasks Are Offloaded

**Critical Mechanism** in `TaskSystem._worker()`:

```python
if asyncio.iscoroutinefunction(handler):
    result = await handler(*args)  # Async: runs in event loop
else:
    # Sync: OFFLOAD TO THREAD POOL
    result = await loop.run_in_executor(None, handler, *args)
```

**Current Usage**:
- ✅ `DirectoryScanner`: Uses dedicated ThreadPoolExecutor for `os.scandir()`
- ⚠️ `ProcessingPipeline`: Handlers are async, but PIL operations block event loop
- ✅ GPU operations: PyTorch releases GIL (no blocking)

### Performance Bottleneck Identified

**Main Issue**: Image preprocessing (PIL operations) runs in event loop:
- `Image.open()` - decoding images (10-50ms)
- `img.thumbnail()` - resizing (5-20ms)
- `img.convert()` - color space conversion (5-10ms)

**Impact**: Blocks other coroutines during Phase 2 processing.

## SAN-14 Implementation Plan

### Phase 1: Quick Wins (1-2 weeks)
**Estimated Impact**: 20-30% faster throughput

1. **Increase Worker Count**: 3 → 8 workers
   ```python
   worker_count = config.get("general.task_workers", 8)  # Was: 3
   ```

2. **Offload PIL Operations**:
   ```python
   # In ThumbnailExtractor
   thumbnail = await loop.run_in_executor(
       None,
       self._generate_thumbnail_sync,  # PIL code here
       file.path
   )
   ```

3. **Add Extractor Flag**:
   ```python
   class Extractor(ABC):
       is_cpu_heavy: bool = False  # New flag
   ```

### Phase 2: Dedicated Executor (2-3 weeks)
**Estimated Impact**: Better resource utilization

4. **Create AI Thread Pool**:
   ```python
   self._ai_executor = ThreadPoolExecutor(
       max_workers=config.get("processing.ai_workers", 4),
       thread_name_prefix="ai-cpu"
   )
   ```

5. **Progress Reporting**:
   ```python
   task_record.progress = int((i + 1) / total * 100)
   await task_record.save()
   await bus.publish("task.progress", {...})
   ```

### Phase 3: Priority Queue (3-4 weeks)
**Estimated Impact**: Significantly improved UX

6. **Priority Queue**:
   ```python
   self._queue = asyncio.PriorityQueue()  # Was: asyncio.Queue
   await self._queue.put((priority, task_id))
   ```

7. **UI Integration**:
   ```python
   # Foreground tasks
   await pipeline.enqueue_phase2(visible_ids, priority=0)  # HIGH
   
   # Background tasks
   await pipeline.enqueue_phase2(background_ids, priority=1)  # NORMAL
   ```

## Integration with Indexer Pipeline

### Complete Flow

```
1. DiscoveryService.scan_root(root_id, background=True)
   ↓
2. TaskSystem.submit("scan_library_root", ...)
   ↓
3. Worker executes DiscoveryService._handle_scan_root()
   - DirectoryScanner.scan_directory_async() [THREAD POOL]
   - DiffDetector.detect_changes() [ASYNC I/O]
   - SyncManager.apply_changes() [ASYNC I/O]
   - ProcessingPipeline.enqueue_phase2() [AUTO-QUEUE]
   ↓
4. TaskSystem.submit("process_phase2_batch", ...)
   ↓
5. Worker executes ProcessingPipeline._handle_phase2_batch()
   - ThumbnailExtractor.process() [⚠️ EVENT LOOP - should be THREAD]
   - MetadataExtractor.process() [OK - C++ releases GIL]
   - CLIPExtractor.process() [⚠️ PIL in EVENT LOOP, GPU OK]
   - XMPExtractor.process() [OK - fast XML parsing]
```

## Task Persistence & Recovery

**Crash Recovery Mechanism**:

```python
# On startup:
# 1. Find interrupted tasks
running_tasks = await TaskRecord.find({"status": "running"})

# 2. Reset to pending
for task in running_tasks:
    task.status = "pending"
    task.error = "Interrupted by system restart"

# 3. Re-queue
pending_tasks = await TaskRecord.find({"status": "pending"})
for task in pending_tasks:
    await self._queue.put(task.id)
```

**Result**: No work is lost on crash. All tasks are automatically retried.

## Performance Analysis

### Current State

| Component | Executor | Speed | Bottleneck |
|-----------|----------|-------|------------|
| DirectoryScanner | Thread Pool | 10K files/sec | Disk I/O ✅ |
| ThumbnailExtractor | Event Loop | ~5-10 files/sec | PIL decoding ⚠️ |
| CLIPExtractor | Event Loop + GPU | ~5-10 files/sec | PIL preprocessing ⚠️ |
| MetadataExtractor | Event Loop (C++) | ~20-30 files/sec | OK ✅ |

### After Optimization (Estimated)

| Component | Executor | Speed | Improvement |
|-----------|----------|-------|-------------|
| ThumbnailExtractor | Thread Pool | ~8-15 files/sec | +30% |
| CLIPExtractor | Thread Pool + GPU | ~8-15 files/sec | +30% |
| Overall Phase 2 | Mixed | ~10-18 files/sec | +20-30% |

## Key Recommendations

### Immediate Actions
1. ✅ **Offload PIL operations** - Highest impact, lowest risk
2. ✅ **Increase worker count** - Simple config change
3. ✅ **Add CPU-heavy flag** - Enables future optimizations

### Future Enhancements
4. Priority queue for foreground tasks
5. Process pool for extreme CPU workloads
6. Distributed task queue for multi-machine setups

## References

- [Technical Documentation](../docs/tasksystem_san14_optimization.md)
- [Session Journal](./journal_session_tasksystem_research.md)
- [TaskSystem Source](../src/core/tasks/system.py)
- [ProcessingPipeline Source](../src/ucorefs/processing/pipeline.py)
- [Indexer Pipeline Architecture](../docs/indexer_pipeline_architecture.md)

---

**Total Estimated Improvement**: 35-55% faster Phase 2 throughput + significantly better UX
