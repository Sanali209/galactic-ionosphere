# TaskSystem Architecture & SAN-14 Optimization Plan

**Document Version**: 1.0  
**Last Updated**: 2025-12-28  
**Related**: [Indexer Pipeline Architecture](indexer_pipeline_architecture.md), [Session Journal](../dev_log/journal_session_tasksystem_research.md)

## Table of Contents

1. [Overview](#overview)
2. [TaskSystem Architecture](#tasksystem-architecture)
3. [CPU-Heavy Task Offloading](#cpu-heavy-task-offloading)
4. [Integration with Indexer Pipeline](#integration-with-indexer-pipeline)
5. [Task Persistence & Recovery](#task-persistence--recovery)
6. [SAN-14: Optimization Opportunities](#san-14-optimization-opportunities)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

The **TaskSystem** is USCore's background task execution engine. It provides:
- **Async worker pool** for concurrent task processing
- **Thread pool offloading** for CPU-heavy operations  
- **Task persistence** for crash recovery
- **Progress tracking** and error handling
- **Integration with indexer pipeline** for file processing

### Key Relationship to Indexer Pipeline

```
DiscoveryService → TaskSystem → ProcessingPipeline
     (scan)       (orchestrate)     (AI processing)
```

The indexer pipeline relies on TaskSystem to:
1. Execute filesystem scans in background
2. Process Phase 2/3 batches without blocking UI
3. Recover from crashes and resume processing

---

## TaskSystem Architecture

### Core Components

**Location**: `src/core/tasks/system.py`

```python
class TaskSystem(BaseSystem):
    """Manages background tasks with persistence and crash recovery."""
    
    # Architecture
    _queue: asyncio.Queue          # In-memory task queue (unbounded)
    _workers: List[asyncio.Task]   # Worker coroutines (default: 3)
    _handlers: Dict[str, Callable] # Registered task handlers
    _running: bool                 # Lifecycle flag
```

### Worker Pool Pattern

```
┌──────────────────────────────────────────┐
│      Main Thread (asyncio event loop)    │
├──────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 ││  ← Async coroutines
│  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│       │             │             │      │
│       ↓             ↓             ↓      │
│  ┌──────────────────────────────────┐   │
│  │   asyncio.Queue (task IDs)       │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
             │
             ↓ (CPU-heavy tasks)
┌──────────────────────────────────────────┐
│   ThreadPoolExecutor (shared/custom)     │
├──────────────────────────────────────────┤
│ ┌─────┐  ┌─────┐  ┌─────┐  ┌───────┐    │
│ │Thr 0│  │Thr 1│  │Thr 2│  │Thr N  │    │  ← OS threads
│ └──┬──┘  └──┬──┘  └──┬──┘  └───┬───┘    │
│    ↓        ↓        ↓          ↓        │
│ scandir  CLIP    EXIF     Image decode   │
└──────────────────────────────────────────┘
```

**Key Points**:
- Workers are **coroutines**, not threads (run in single thread)
- They yield control during I/O operations (`await`)
- For CPU-heavy work, they offload to **thread pool**

### Initialization & Recovery

```python
async def initialize(self):
    # 1. CRASH RECOVERY: Reset interrupted tasks
    running_tasks = await TaskRecord.find({"status": "running"})
    for task in running_tasks:
        task.status = "pending"  # Will be retried
        task.error = "Interrupted by system restart"
        await task.save()
    
    # 2. RELOAD PENDING: Queue all pending tasks
    pending_tasks = await TaskRecord.find({"status": "pending"})
    for task in pending_tasks:
        await self._queue.put(task.id)
    
    # 3. START WORKERS: Configurable count
    worker_count = config.get("general.task_workers", 3)
    for i in range(worker_count):
        task = asyncio.create_task(self._worker(i))
        self._workers.append(task)
```

**Recovery Benefit**: No work is lost on crash. All pending/interrupted tasks are automatically retried.

### Handler Registration

Services register their task handlers at initialization:

```python
# ProcessingPipeline registers Phase 2/3 handlers
task_system.register_handler("process_phase2_batch", self._handle_phase2_batch)
task_system.register_handler("process_phase3_item", self._handle_phase3_item)

# DiscoveryService registers scan handler
task_system.register_handler("scan_library_root", self._handle_scan_root)
```

### Task Submission Flow

```
1. Service: task_id = await task_system.submit(handler_name, task_name, *args)
   ↓
2. TaskSystem: Create TaskRecord and save to MongoDB
   {
       name: "Phase 2: Process 20 files",
       handler_name: "process_phase2_batch",
       task_args: ["file_id_1,file_id_2,..."],
       status: "pending",
       created_at: 1735372800
   }
   ↓
3. TaskSystem: Put task ID in queue
   await self._queue.put(task_record.id)
   ↓
4. Worker: Pick task from queue
   task_id = await self._queue.get()
   ↓
5. Worker: Load TaskRecord and execute handler
   record = await TaskRecord.get(task_id)
   handler = self._handlers[record.handler_name]
   result = await handler(*record.task_args)
   ↓
6. Worker: Save result
   record.status = "completed"
   record.result = str(result)
   await record.save()
```

---

## CPU-Heavy Task Offloading

### The Critical Mechanism

**Location**: `TaskSystem._worker()` (line 131-144)

```python
handler = self._handlers.get(record.handler_name)

if asyncio.iscoroutinefunction(handler):
    # Async handler: runs in event loop (I/O-bound)
    result = await handler(*record.task_args)
else:
    # Sync handler: OFFLOAD TO THREAD POOL (CPU-bound)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,  # Use default ThreadPoolExecutor
        partial(handler, *record.task_args)
    )
```

**How it Works**:
1. **Async handlers**: Execute in event loop, yield during I/O
2. **Sync handlers**: Automatically offloaded to thread pool
3. **Default pool**: Python's `ThreadPoolExecutor` with `min(32, cpu_count + 4)` workers

### Example: DirectoryScanner

**Problem**: `os.scandir()` is a **blocking** system call (can take minutes for large directories)

**Solution**: Run in dedicated thread pool

```python
# In DirectoryScanner.scan_directory_async()
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scanner") as executor:
    def collect_results():
        """Run in thread: collect all scan results."""
        return list(self.scan_directory(...))  # Blocking os.scandir()
    
    # Offload to thread
    batches = await loop.run_in_executor(executor, collect_results)
    
    # Yield batches back to async context
    for batch in batches:
        yield batch
        await asyncio.sleep(0)  # Let event loop breathe
```

**Key Benefits**:
- UI remains responsive during scans
- Uses dedicated executor (not shared with other tasks)
- Results returned incrementally (batches)

### Current State: AI Extractors

**ProcessingPipeline Phase 2/3 handlers are async**:

```python
async def _handle_phase2_batch(self, file_ids_str: str):
    files = [await FileRecord.get(fid) for fid in file_ids]  # Async I/O
    
    extractors = ExtractorRegistry.get_for_phase(2, locator=self.locator)
    
    for extractor in extractors:
        # Extractor.process() is async
        results = await extractor.process(processable)  # ← AI inference here
```

**Potential Issue**: If `extractor.process()` does CPU-heavy work synchronously, it will block the event loop.

**Current Mitigation**:
- Most AI models (CLIP, BLIP) use PyTorch with **GPU acceleration**
- GPU operations release the GIL (Python Global Interpreter Lock)
- So they don't actually block other coroutines

**Remaining Bottleneck**: CPU preprocessing (image decoding, resizing) still happens in event loop.

---

## Integration with Indexer Pipeline

### Complete Flow: Discovery → Processing

```
1. User Action: Add library root
   ↓
2. DiscoveryService.scan_root(root_id, background=True)
   ↓
3. Submit to TaskSystem
   task_id = await task_system.submit("scan_library_root", "Scan root", str(root_id))
   ↓
4. Worker executes DiscoveryService._handle_scan_root()
   
   async def _handle_scan_root(self, root_id_str: str):
       # Load root
       root = await DirectoryRecord.get(ObjectId(root_id_str))
       
       # Scan filesystem (offloaded to thread pool)
       for batch in scanner.scan_directory_async(root.path, ...):
           # Detect changes
           diff = await diff_detector.detect_changes(batch, incremental=True)
           
           # Apply changes
           stats = await sync_manager.apply_changes(diff, str(root._id))
           
           # AUTO-QUEUE PHASE 2
           added_ids = stats.get("added_file_ids", [])
           if added_ids and self.processing_pipeline:
               await self.processing_pipeline.enqueue_phase2(added_ids)
   ↓
5. ProcessingPipeline.enqueue_phase2(file_ids)
   
   # Batch into groups of 20
   for i in range(0, len(file_ids), 20):
       batch = file_ids[i:i+20]
       task_id = await task_system.submit(
           "process_phase2_batch",
           f"Phase 2: Process {len(batch)} files",
           ",".join(str(fid) for fid in batch)
       )
   ↓
6. Worker executes ProcessingPipeline._handle_phase2_batch()
   
   # Load files
   files = [await FileRecord.get(fid) for fid in file_ids]
   
   # Get Phase 2 extractors
   extractors = ExtractorRegistry.get_for_phase(2, locator=self.locator)
   
   # Run each extractor
   for extractor in extractors:  # Thumbnail, Metadata, CLIP, XMP
       processable = [f for f in files if extractor.can_process(f)]
       results = await extractor.process(processable)
```

### Task Handlers Summary

| Handler | Registered By | Purpose | Execution |
|---------|---------------|---------|-----------|
| `scan_library_root` | DiscoveryService | Scan filesystem | Offloads `os.scandir()` to thread |
| `process_phase2_batch` | ProcessingPipeline | Metadata + basic AI | Async (GPU accelerated) |
| `process_phase3_item` | ProcessingPipeline | Advanced AI | Async (GPU accelerated) |

---

## Task Persistence & Recovery

### TaskRecord Model

**Location**: `src/core/tasks/models.py`

```python
class TaskRecord(CollectionRecord):
    _collection_name = "tasks"
    
    name = StringField()              # "Phase 2: Process 20 files"
    handler_name = StringField()      # "process_phase2_batch"
    task_args = ListField(StringField())  # ["file_id_1,file_id_2,..."]
    
    status = StringField(default="pending")  # pending | running | completed | failed
    progress = IntField(default=0)    # 0-100
    
    result = StringField()            # Serialized result (JSON string)
    error = StringField()             # Error message if failed
    created_at = IntField()           # Unix timestamp
```

### Recovery Process

**On Application Startup**:

```python
# 1. Find interrupted tasks (status="running")
running_tasks = await TaskRecord.find({"status": "running"})

if running_tasks:
    logger.warning(f"Found {len(running_tasks)} interrupted tasks")
    
    for task in running_tasks:
        task.status = "pending"  # Reset to pending
        task.error = "Interrupted by system restart"
        await task.save()

# 2. Reload ALL pending tasks
pending_tasks = await TaskRecord.find({"status": "pending"})

for task in pending_tasks:
    await self._queue.put(task.id)  # Re-queue for processing

logger.info(f"Loaded {len(pending_tasks)} pending tasks")
```

**Result**: All work is resumed automatically after crash/restart.

### Error Handling & Exponential Backoff

**Location**: `TaskSystem._worker()` (line 160-178)

```python
consecutive_failures = 0
max_consecutive_failures = 5
base_backoff_seconds = 1.0

while self._running:
    try:
        # Process task...
        consecutive_failures = 0  # Reset on success
        
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
        consecutive_failures += 1
        
        if consecutive_failures >= max_consecutive_failures:
            logger.critical(f"Worker hit max failures ({max_consecutive_failures})")
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
        backoff = min(base_backoff_seconds * (2 ** (consecutive_failures - 1)), 30.0)
        
        await asyncio.sleep(backoff)
```

**Why Exponential Backoff?**
- Prevents "thundering herd" (all workers retrying simultaneously)
- Gives time for transient issues to resolve (network, database)
- Reduces resource contention during outages

---

## SAN-14: Optimization Opportunities

### Current Performance Analysis

#### What's Already Optimized ✅

1. **Async Workers**: Efficient for I/O-bound tasks
2. **Thread Pool Fallback**: Sync handlers automatically offloaded
3. **DirectoryScanner**: Uses dedicated thread pool for filesystem I/O
4. **Crash Recovery**: Tasks persisted to MongoDB
5. **Exponential Backoff**: Prevents runaway failures
6. **GPU Acceleration**: AI models release GIL during inference

#### Identified Bottlenecks ⚠️

| Component | Current State | Bottleneck | Impact |
|-----------|---------------|------------|--------|
| **ThumbnailExtractor** | Async (event loop) | PIL image decoding/resizing | Medium (blocks event loop ~10-50ms per file) |
| **MetadataExtractor** | Async (event loop) | EXIF parsing (pyexiv2) | Low (C++ library releases GIL) |
| **CLIPExtractor** | Async (event loop) | Image preprocessing (CPU) | Medium (PIL operations block) |
| **XMPExtractor** | Async (event loop) | XML parsing | Low (fast) |

**Key Finding**: The main optimization opportunity is **image preprocessing** (decoding, resizing) which happens in PIL before GPU inference.

### Proposed Optimizations

#### 1. Offload Image Operations to Thread Pool

**Target**: `ThumbnailExtractor`, `CLIPExtractor`

**Implementation**:

```python
class ThumbnailExtractor(Extractor):
    async def process(self, files: List[FileRecord]) -> Dict[ObjectId, bool]:
        loop = asyncio.get_event_loop()
        results = {}
        
        for file in files:
            # Offload PIL operations to thread pool
            thumbnail = await loop.run_in_executor(
                None,  # Default executor
                self._generate_thumbnail_sync,  # Blocking function
                file.path
            )
            
            if thumbnail:
                # Save (async I/O)
                await thumbnail_service.save(file._id, thumbnail)
                results[file._id] = True
            else:
                results[file._id] = False
        
        return results
    
    def _generate_thumbnail_sync(self, path: str):
        """Blocking: runs in thread pool."""
        from PIL import Image
        
        img = Image.open(path)
        img.thumbnail((256, 256))
        return img
```

**Benefit**: PIL operations won't block event loop, allowing other coroutines to run.

**Estimated Impact**: ~20-30% faster Phase 2 throughput for image-heavy collections.

#### 2. Increase Default Worker Count

**Current**: 3 workers (default)

**Proposed**: `min(8, os.cpu_count())`

**Rationale**:
- Modern systems have 4-16 cores
- Workers are lightweight (coroutines, not threads)
- More workers = better concurrency for I/O-bound tasks

**Configuration**:

```json
{
  "general": {
    "task_workers": 8
  }
}
```

#### 3. Dedicated AI Thread Pool

**Create separate executor for AI preprocessing**:

```python
class ProcessingPipeline(BaseSystem):
    async def initialize(self):
        super().initialize()
        
        # Dedicated pool for CPU-heavy image operations
        ai_workers = config.get("processing.ai_workers", 4)
        self._ai_executor = ThreadPoolExecutor(
            max_workers=ai_workers,
            thread_name_prefix="ai-cpu"
        )
    
    async def shutdown(self):
        self._ai_executor.shutdown(wait=True)
        await super().shutdown()
```

**Usage in extractors**:

```python
# In CLIPExtractor
async def process(self, files):
    # Preprocessing in dedicated pool
    images = await loop.run_in_executor(
        pipeline._ai_executor,  # Use dedicated pool
        self._preprocess_images_sync,
        [f.path for f in files]
    )
    
    # Inference on GPU (async, releases GIL)
    embeddings = await self._encode_images_gpu(images)
```

#### 4. Priority Queue for Foreground Tasks

**Replace `asyncio.Queue` with `asyncio.PriorityQueue`**:

```python
class TaskSystem(BaseSystem):
    def __init__(self, locator, config):
        self._queue = asyncio.PriorityQueue()  # ← Priority queue
        # ...
    
    async def submit(self, handler_name, task_name, *args, priority=1):
        # Default priority=1 (normal)
        # UI-visible files get priority=0 (high)
        
        record = TaskRecord(...)
        await record.save()
        
        await self._queue.put((priority, record.id))  # ← Tuple: (priority, task_id)
```

**UI Integration**:

```python
# In FilePaneWidget
visible_file_ids = self.get_visible_file_ids()

# Queue with high priority
await pipeline.enqueue_phase2(visible_file_ids, priority=0)  # HIGH

# Background files (low priority)
await pipeline.enqueue_phase2(background_ids, priority=1)  # NORMAL
```

**Benefit**: Files visible in UI are processed first, improving perceived responsiveness.

#### 5. Progress Reporting for Long Tasks

**Add progress callbacks**:

```python
class ProcessingPipeline:
    async def _handle_phase2_batch(self, file_ids_str: str):
        file_ids = [ObjectId(fid) for fid in file_ids_str.split(",")]
        total = len(file_ids)
        
        for i, file_id in enumerate(file_ids):
            # Process file...
            
            # Update progress
            progress = int((i + 1) / total * 100)
            task_record = await TaskRecord.get(current_task_id)
            task_record.progress = progress
            await task_record.save()
            
            # Publish event for UI
            await bus.publish("task.progress", {
                "task_id": current_task_id,
                "progress": progress
            })
```

**UI Integration** (BackgroundPanel):

```python
# Subscribe to progress events
bus.subscribe("task.progress", self._on_task_progress)

def _on_task_progress(self, event):
    task_id = event["task_id"]
    progress = event["progress"]
    
    # Update progress bar in UI
    self.update_task_progress(task_id, progress)
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**Priority**: High  
**Risk**: Low

1. **Increase Worker Count**
   - Change default from 3 to 8
   - Add config validation
   - Document in configuration guide

2. **Offload Image Decoding**
   - Wrap PIL operations in `run_in_executor()`
   - Target: `ThumbnailExtractor`, `CLIPExtractor`

3. **Add Extractor Flag**
   - Add `is_cpu_heavy` property to `Extractor` base class
   - Mark: Thumbnail, Metadata (EXIF)

**Estimated Impact**: 20-30% faster Phase 2 throughput

### Phase 2: Dedicated Executor (2-3 weeks)

**Priority**: Medium  
**Risk**: Low

4. **Create AI Thread Pool**
   - Add `_ai_executor` to `ProcessingPipeline`
   - Configurable size: `config.processing.ai_workers`
   - Use in CPU-heavy extractors

5. **Progress Reporting**
   - Implement progress callbacks in handlers
   - Update `TaskRecord.progress` during execution
   - Publish `task.progress` events

**Estimated Impact**: Better resource utilization, improved user feedback

### Phase 3: Priority Queue (3-4 weeks)

**Priority**: Medium  
**Risk**: Medium (requires UI changes)

6. **Priority Queue Implementation**
   - Replace `asyncio.Queue` with `PriorityQueue`
   - Add `priority` parameter to `submit()`
   - Update all callers

7. **UI Integration**
   - Detect visible file IDs in FilePaneWidget
   - Submit visible files with HIGH priority
   - Background files with NORMAL priority

**Estimated Impact**: Significantly improved perceived responsiveness

### Phase 4: Advanced (Future)

**Priority**: Low  
**Risk**: High

8. **Process Pool for True Parallelism**
   - Use `ProcessPoolExecutor` for CPU-bound tasks
   - Only for tasks without shared state
   - Requires pickling overhead analysis

9. **Distributed Task Queue**
   - Replace in-memory queue with Redis/RabbitMQ
   - Enable multi-machine processing
   - Requires infrastructure changes

10. **GPU Task Scheduling**
    - Detect GPU count and availability
    - Schedule tasks across multiple GPUs
    - Use CUDA streams for concurrency

---

## Summary

### Key Findings

1. **TaskSystem is well-architected** for I/O-bound tasks
2. **Main bottleneck** is CPU preprocessing (image decoding) in event loop
3. **GPU acceleration** already works well (releases GIL)
4. **Quick wins available** (offload PIL operations, increase workers)
5. **Integration with indexer** is clean and effective

### Recommended Actions for SAN-14

**Short-Term**:
- ✅ Offload image operations to thread pool
- ✅ Increase default worker count to 8
- ✅ Add `is_cpu_heavy` flag to extractors

**Medium-Term**:
- ✅ Create dedicated AI thread pool
- ✅ Implement progress reporting
- ✅ Priority queue for foreground tasks

**Long-Term**:
- Consider process pool for extreme CPU workloads
- Distributed queue for multi-machine setups

### Performance Gains Expected

| Optimization | Difficulty | Impact | Timeline |
|--------------|------------|--------|----------|
| Offload image ops | Low | 20-30% | 1 week |
| Increase workers | Low | 10-15% | 1 day |
| Dedicated executor | Medium | 5-10% | 2 weeks |
| Priority queue | Medium | UX improvement | 3 weeks |

**Total Estimated Improvement**: 35-55% faster Phase 2 throughput + better UX

---

## References

- [TaskSystem Implementation](file:///d:/github/USCore/src/core/tasks/system.py)
- [TaskRecord Model](file:///d:/github/USCore/src/core/tasks/models.py)
- [ProcessingPipeline](file:///d:/github/USCore/src/ucorefs/processing/pipeline.py)
- [DirectoryScanner](file:///d:/github/USCore/src/ucorefs/discovery/scanner.py)
- [Indexer Pipeline Architecture](indexer_pipeline_architecture.md)
- [Session Research Journal](../dev_log/journal_session_tasksystem_research.md)

---

**Document End**
