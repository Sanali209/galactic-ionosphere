# TaskSystem Architecture Research - Session Journal

**Date**: 2025-12-28  
**Session ID**: tasksystem_architecture_research  
**Status**: Complete

## Objective

Deep research on TaskSystem architecture and its relationship to the indexer pipeline, focusing on:
- How TaskSystem offloads CPU-heavy tasks to prevent UI blocking
- Integration with ProcessingPipeline and DiscoveryService
- Thread pool executor usage patterns
- Task persistence and crash recovery

## [PROGRESS]
- Phase: Analysis | Execution
- Step: Comprehensive documentation completed
- Completed: 12/12 steps
- Next: Create technical documentation
[/PROGRESS]

## Research Findings

###1. TaskSystem Core Architecture

**Location**: `src/core/tasks/system.py`

#### Key Components

```python
class TaskSystem(BaseSystem):
    """Manages background tasks with persistence and crash recovery."""
    
    def __init__(self, locator, config):
        self._queue = asyncio.Queue()          # In-memory task queue
        self._workers = []                     # Worker coroutines
        self._running = False                  # Lifecycle flag
        self._handlers: Dict[str, Callable] = {}  # Registered handlers
```

#### Worker Pool Pattern

```python
async def initialize(self):
    # 1. Crash Recovery
    running_tasks = await TaskRecord.find({"status": "running"})
    for task in running_tasks:
        task.status = "pending"  # Reschedule interrupted tasks
        task.error = "Interrupted by system restart"
    
    # 2. Reload Pending Tasks
    pending_tasks = await TaskRecord.find({"status": "pending"})
    for task in pending_tasks:
        await self._queue.put(task.id)  # Re-queue
    
    # 3. Start Workers (configurable count)
    worker_count = config.get("general.task_workers", 3)  # Default: 3
    
    for i in range(worker_count):
        task = asyncio.create_task(self._worker(i))
        self._workers.append(task)
```

**Key Insight**: Workers are **async coroutines**, NOT threads. They run in the same process but yield control during I/O operations.

#### Handler Registration Pattern

**ProcessingPipeline** registers handlers:

```python
# In ProcessingPipeline.initialize()
self.task_system.register_handler("process_phase2_batch", self._handle_phase2_batch)
self.task_system.register_handler("process_phase3_item", self._handle_phase3_item)
```

**DiscoveryService** registers handlers:

```python
# In DiscoveryService.initialize()
self.task_system.register_handler("scan_library_root", self._handle_scan_root)
```

#### Task Submission Flow

```
1. Service calls task_system.submit(handler_name, task_name, *args)
   ↓
2. TaskRecord created and saved to MongoDB
   {
       name: "Phase 2: Process 20 files",
       handler_name: "process_phase2_batch",
       task_args: ["file_id_1,file_id_2,..."],
       status: "pending"
   }
   ↓
3. Task ID added to asyncio.Queue
   ↓
4. Worker picks task from queue
   ↓
5. Worker executes registered handler
   ↓
6. Result saved to TaskRecord, status → "completed" / "failed"
```

### 2. CPU-Heavy Task Offloading Mechanism

#### Critical Code Section

```python
# In TaskSystem._worker()
handler = self._handlers.get(record.handler_name)

if asyncio.iscoroutinefunction(handler):
    # Async handler: runs in event loop
    result = await handler(*record.task_args)
else:
    # Synchronous handler: OFFLOAD TO THREAD POOL
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,  # Use default ThreadPoolExecutor
        partial(handler, *record.task_args)
    )
```

**Key Mechanism**: `loop.run_in_executor(None, func)` uses Python's **default ThreadPoolExecutor**:
- Default pool size: `min(32, os.cpu_count() + 4)` workers
- Executor is shared across all `run_in_executor()` calls
- Threads are reused (pool pattern)

#### Where CPU-Heavy Tasks Are Offloaded

**1. DirectoryScanner (Filesystem I/O)**

```python
# In DirectoryScanner.scan_directory_async()
with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scanner") as executor:
    def collect_results():
        """Run in thread: collect all scan results."""
        return list(self.scan_directory(...))  # Blocking os.scandir()
    
    batches = await loop.run_in_executor(executor, collect_results)
```

**Why thread pool?**
- `os.scandir()` is **synchronous** (blocks)
- Can take seconds/minutes for large directories
- Running in thread prevents UI freeze

**Note**: Uses **dedicated executor** (max_workers=1) instead of default pool.

**2. ProcessingPipeline Handlers**

Currently, Phase 2/3 handlers are **async** (not offloaded):

```python
async def _handle_phase2_batch(self, file_ids_str: str) -> dict:
    # Load files from DB (async I/O)
    files = [await FileRecord.get(fid) for fid in file_ids]
    
    # Get extractors
    extractors = ExtractorRegistry.get_for_phase(2, locator=self.locator)
    
    # Run extractors (currently async, but COULD be CPU-heavy)
    for extractor in extractors:
        results = await extractor.process(processable)  # ← AI inference here
```

**Potential Issue**: If extractors do CPU-heavy work (CLIP inference, image processing) in the main thread, they will block the event loop.

**Current Mitigation**: Most extractors use PyTorch with GPU, which releases GIL during inference.

**Better Solution**: Wrap CPU-heavy operations in `run_in_executor()`.

### 3. Integration with Indexer Pipeline

#### Data Flow: DiscoveryService → TaskSystem → ProcessingPipeline

```
1. DiscoveryService.scan_root(root_id, background=True)
   ↓
2. Submits "scan_library_root" task
   task_id = await task_system.submit("scan_library_root", "Scan root", str(root_id))
   ↓
3. Worker executes DiscoveryService._handle_scan_root()
   async def _handle_scan_root(self, root_id_str: str):
       # Offloaded to thread pool via DirectoryScanner
       for batch in scanner.scan_directory_async(...):
           diff = await diff_detector.detect_changes(batch)
           stats = await sync_manager.apply_changes(diff)
           
           # Auto-queue Phase 2
           added_ids = stats["added_file_ids"]
           await processing_pipeline.enqueue_phase2(added_ids)
   ↓
4. ProcessingPipeline.enqueue_phase2(file_ids)
   ↓
5. Submits "process_phase2_batch" tasks (batch size: 20)
   for i in range(0, len(file_ids), 20):
       batch = file_ids[i:i+20]
       task_id = await task_system.submit("process_phase2_batch", ...)
   ↓
6. Worker executes ProcessingPipeline._handle_phase2_batch()
   for extractor in extractors:
       results = await extractor.process(files)  # CLIP, thumbnails, etc.
```

#### Execution Model

```
┌──────────────────────────────────────────────────────────┐
│              Main Thread (asyncio event loop)            │
├──────────────────────────────────────────────────────────┤
│ ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│ │ Worker 0 │   │ Worker 1 │   │ Worker 2 │  (coroutines)│
│ └────┬─────┘   └────┬─────┘   └────┬─────┘              │
│      │              │              │                     │
│      ↓              ↓              ↓                     │
│ ┌────────────────────────────────────────┐               │
│ │     asyncio.Queue (task_ids)           │               │
│ └────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
                       │
                       ↓ (CPU-heavy tasks)
┌──────────────────────────────────────────────────────────┐
│        ThreadPoolExecutor (default or custom)            │
├──────────────────────────────────────────────────────────┤
│ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│ │Thread 0 │  │Thread 1 │  │Thread 2 │  │Thread N │     │
│ └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │
│      │            │            │            │           │
│      ↓            ↓            ↓            ↓           │
│  os.scandir()  CLIP model  Image decode  EXIF parse    │
└──────────────────────────────────────────────────────────┘
```

### 4. Task Persistence & Crash Recovery

#### TaskRecord Model

**Location**: `src/core/tasks/models.py`

```python
class TaskRecord(CollectionRecord):
    _collection_name = "tasks"
    
    name = StringField()              # Human-readable title
    handler_name = StringField()      # Registered handler key
    task_args = ListField(StringField())  # Serialized arguments
    
    status = StringField(default="pending")  # pending | running | completed | failed
    progress = IntField(default=0)    # 0-100
    
    result = StringField()            # Serialized result
    error = StringField()             # Error message if failed
    created_at = IntField()           # Unix timestamp
```

#### Recovery Mechanism

On startup:

```python
# 1. Find tasks that were "running" when app crashed
running_tasks = await TaskRecord.find({"status": "running"})

# 2. Reset to "pending" (will be retried)
for task in running_tasks:
    task.status = "pending"
    task.error = "Interrupted by system restart"
    await task.save()

# 3. Reload ALL pending tasks
pending_tasks = await TaskRecord.find({"status": "pending"})
for task in pending_tasks:
    await self._queue.put(task.id)  # Re-queue
```

**Key Benefit**: No work is lost on crash. Tasks are retried automatically.

#### Error Handling & Backoff

```python
# In TaskSystem._worker()
consecutive_failures = 0
max_consecutive_failures = 5
base_backoff_seconds = 1.0

while self._running:
    try:
        # Execute task...
        consecutive_failures = 0  # Reset on success
    
    except Exception as e:
        consecutive_failures += 1
        
        if consecutive_failures >= max_consecutive_failures:
            logger.critical(f"Worker hit max failures ({max_consecutive_failures})")
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
        backoff = min(base_backoff_seconds * (2 ** (consecutive_failures - 1)), 30.0)
        await asyncio.sleep(backoff)
```

**Pattern**: Exponential backoff prevents thundering herd on persistent failures.

### 5. Relationship to SAN-14: Task Execution Optimization

#### Current State Analysis

**What's Already Optimized**:
1. ✅ **Async Workers**: Workers are coroutines, not threads (efficient for I/O)
2. ✅ **Thread Pool Fallback**: Sync handlers automatically offloaded
3. ✅ **DirectoryScanner**: Uses dedicated thread pool for `os.scandir()`
4. ✅ **Crash Recovery**: Tasks persisted to MongoDB
5. ✅ **Exponential Backoff**: Prevents runaway failures

**What Could Be Improved (SAN-14)**:

1. **AI Extractor CPU Usage**: 
   - CLIP, BLIP, GroundingDINO models may block event loop
   - **Fix**: Wrap inference in `run_in_executor()`

2. **Image Decoding**:
   - `PIL.Image.open()` can be slow for large images
   - **Fix**: Offload to thread pool

3. **EXIF Parsing**:
   - Can be CPU-intensive for RAW files
   - **Fix**: Already using pyexiv2 which is C++ (releases GIL), but could wrap

4. **Configurable Thread Pool Size**:
   - Default pool size may not be optimal
   - **Fix**: Add config option for custom executor with tunable max_workers

#### Proposed Optimizations

**Option 1: Dedicated AI Thread Pool**

```python
class ProcessingPipeline(BaseSystem):
    async def initialize(self):
        # Create dedicated pool for CPU-heavy AI tasks
        self._ai_executor = ThreadPoolExecutor(
            max_workers=config.get("processing.ai_workers", 4),
            thread_name_prefix="ai-worker"
        )
    
    async def _handle_phase2_batch(self, file_ids_str: str):
        # ... load files ...
        
        for extractor in extractors:
            if extractor.is_cpu_heavy:  # New flag
                # Offload to thread pool
                result = await loop.run_in_executor(
                    self._ai_executor,
                    extractor.process_sync,  # Blocking version
                    processable
                )
            else:
                # Keep async for I/O-bound extractors
                result = await extractor.process(processable)
```

**Option 2: Process Pool for True Parallelism**

```python
from concurrent.futures import ProcessPoolExecutor

class ProcessingPipeline(BaseSystem):
    async def initialize(self):
        # Use multiprocessing for GIL-free parallelism
        self._ai_executor = ProcessPoolExecutor(
            max_workers=config.get("processing.ai_processes", 2)
        )
```

**Trade-off**: Process pool has higher overhead (pickling data), but achieves true parallelism.

**Option 3: Hybrid Approach**

- **Thread pool** for I/O-bound tasks (EXIF parsing, image decoding)
- **Process pool** for CPU-bound tasks (AI inference)
- **Async** for network/database I/O

### 6. Performance Characteristics

#### Current TaskSystem Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Worker Count | 3 (default) | Configurable via `config.general.task_workers` |
| Queue Type | `asyncio.Queue` | In-memory, unbounded |
| Persistence | MongoDB | All tasks persisted |
| Crash Recovery | Automatic | Tasks re-queued on startup |
| Backoff Strategy | Exponential | 1s → 2s → 4s → 8s → 16s (max 30s) |
| Thread Pool | Default | `min(32, cpu_count + 4)` threads |

#### Indexer Pipeline Performance

| Phase | Current Executor | Bottleneck | Optimization Opportunity |
|-------|------------------|------------|--------------------------|
| Phase 1 (Discovery) | Thread Pool (1 worker) | Disk I/O | ✅ Already optimized |
| Phase 2 (CLIP) | Async (event loop) | **GPU inference (OK)**, CPU pre-processing | Offload image decoding |
| Phase 2 (Metadata) | Async (event loop) | **EXIF parsing** | Offload to thread pool |
| Phase 2 (Thumbnails) | Async (event loop) | Image resizing | Offload to thread pool |
| Phase 3 (BLIP) | Async (event loop) | GPU inference (OK) | Already optimal (GPU releases GIL) |
| Phase 3 (Detection) | Async (event loop) | GPU inference (OK) | Already optimal |

**Key Finding**: Most CPU-heavy work (AI inference) is already GPU-accelerated, which releases the GIL. The main optimization opportunity is **image decoding and preprocessing**.

### 7. Comparison: TaskSystem vs Direct Threading

#### Current: TaskSystem Approach

**Pros**:
- ✅ Task persistence (crash recovery)
- ✅ Progress tracking
- ✅ Centralized error handling
- ✅ Easy to monitor (BackgroundPanel in UI)
- ✅ Configurable worker count
- ✅ Automatic retry with backoff

**Cons**:
- ⚠️ Overhead of MongoDB writes per task
- ⚠️ All tasks share same worker pool (no prioritization)
- ⚠️ Limited to 3 workers by default (though configurable)

#### Alternative: Direct `run_in_executor()`

**Example**:

```python
# Direct approach (no TaskSystem)
loop = asyncio.get_event_loop()
results = await loop.run_in_executor(
    None,  # Default thread pool
    blocking_function,
    args
)
```

**Pros**:
- ✅ Lower overhead (no MongoDB writes)
- ✅ Direct control over executor
- ✅ Simpler code

**Cons**:
- ❌ No crash recovery
- ❌ No progress tracking
- ❌ No centralized error handling
- ❌ Hard to monitor

### 8. Recommendations for SAN-14

Based on research, here are specific recommendations:

#### Short-Term (Quick Wins)

1. **Identify CPU-Heavy Extractors**
   - Add `is_cpu_heavy` flag to `Extractor` base class
   - Mark: `ThumbnailExtractor`, `MetadataExtractor` (EXIF)

2. **Wrap Image Operations**
   ```python
   # In ThumbnailExtractor
   async def process(self, files):
       loop = asyncio.get_event_loop()
       
       for file in files:
           thumbnail = await loop.run_in_executor(
               None,
               self._generate_thumbnail_sync,  # Blocking PIL operations
               file.path
           )
   ```

3. **Increase Default Worker Count**
   - Change default from 3 to `min(8, os.cpu_count())`
   - Rationale: Modern systems can handle more concurrent tasks

#### Medium-Term (Phase 5 Roadmap)

4. **Priority Queue Implementation**
   - Replace `asyncio.Queue` with `asyncio.PriorityQueue`
   - Add priority levels: HIGH (foreground), NORMAL (background)
   - UI-visible files get HIGH priority

5. **Dedicated AI Executor**
   - Create separate `ThreadPoolExecutor` for AI tasks
   - Configurable size: `config.processing.ai_workers`

6. **Progress Callbacks**
   - Add progress reporting to long-running tasks
   - Update `TaskRecord.progress` field during execution

#### Long-Term (Future)

7. **Process Pool for CPU-Bound Work**
   - Use `ProcessPoolExecutor` for true parallelism
   - Only for tasks that don't need shared state

8. **Distributed Task Queue**
   - Replace in-memory queue with Redis/RabbitMQ
   - Enable multi-machine processing

9. **GPU Task Scheduling**
   - Detect GPU count and availability
   - Schedule GPU tasks across multiple GPUs
   - Use CUDA streams for concurrent execution

## Next Steps

1. ✅ Document TaskSystem architecture
2. ✅ Identify optimization opportunities
3. [ ] Create implementation plan for SAN-14
4. [ ] Add to roadmap Phase 2

## References

- [TaskSystem](file:///d:/github/USCore/src/core/tasks/system.py)
- [TaskRecord Model](file:///d:/github/USCore/src/core/tasks/models.py)
- [ProcessingPipeline](file:///d:/github/USCore/src/ucorefs/processing/pipeline.py)
- [DirectoryScanner](file:///d:/github/USCore/src/ucorefs/discovery/scanner.py)
- [Indexer Pipeline Architecture](../docs/indexer_pipeline_architecture.md)

---

**Session End Time**: 2025-12-28 10:15:00  
**Duration**: ~15 minutes  
**Output**: Comprehensive TaskSystem documentation and SAN-14 optimization plan
