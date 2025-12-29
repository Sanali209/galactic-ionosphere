# Session Journal: Indexer Pipeline Research
**Date**: 2025-12-28
**Session ID**: indexer_pipeline_research
**Status**: In Progress

## Objective
Conduct deep research on the indexer pipeline architecture, how it interacts with other system components, understand the implementation in ucorefs, and document how it's used in uexplorer.

## [PROGRESS]
- Phase: Analysis
- Step: Analyzing core pipeline components and data flow
- Completed: 8/12 steps
- Next: Create comprehensive documentation in docs/ directory
[/PROGRESS]

## Research Overview

### 1. Core Components Identified

#### A. Discovery System (`src/ucorefs/discovery/`)
- **DirectoryScanner**: Scans filesystem directories in batches (200 items default)
- **DiffDetector**: Detects changes between filesystem and database state
- **SyncManager**: Applies detected changes to database atomically
- **LibraryManager**: Manages library roots, watch lists, and blacklists
- **DiscoveryService**: Orchestrates the entire discovery process

#### B. Processing Pipeline (`src/ucorefs/processing/pipeline.py`)
- **Phase 2 Processing**: Thumbnails, metadata extraction, basic embeddings (batch size: 20)
- **Phase 3 Processing**: AI analysis, BLIP captioning, object detection (batch size: 1)
- Task queue management via TaskSystem
- Event-driven architecture using CommandBus

#### C. Extractor Registry (`src/ucorefs/extractors/`)
- Plugin architecture for file processing
- **Phase 2 Extractors**: ThumbnailExtractor, MetadataExtractor, CLIPExtractor, XMPExtractor
- **Phase 3 Extractors**: BLIPExtractor, GroundingDINOExtractor, WDTaggerExtractor
- **ExtractorRegistry**: Central registry managing all extractors

#### D. Vector Services (`src/ucorefs/vectors/`)
- **FAISSIndexService**: In-memory vector similarity search
- **VectorService**: Wrapper service (being deprecated in favor of SearchService)
- **EmbeddingRecord**: Stores vectors in MongoDB

#### E. Search Integration (`src/ucorefs/search/service.py`)
- **SearchService**: Unified search combining MongoDB filters and FAISS vector search
- Supports text, metadata, and vector similarity search
- Result scoring and ranking

## 2. Indexer Pipeline Flow

### Complete Pipeline Architecture

```
┌──────────────────┐
│  User Action     │
│  (Add Library    │
│   Root)          │
└────────┬─────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: DISCOVERY                         │
├─────────────────────────────────────────────────────────┤
│ 1. DirectoryScanner.scan_directory_async()             │
│    - Runs in thread pool (non-blocking I/O)            │
│    - Yields batches of 200 items                       │
│    - Respects watch_extensions & blacklist_paths       │
│                                                         │
│ 2. DiffDetector.detect_changes()                       │
│    - Compares scan results with DB state               │
│    - Detects: added, modified, deleted files/dirs      │
│    - Incremental mode (skip deletions in batches)      │
│                                                         │
│ 3. SyncManager.apply_changes()                         │
│    - Creates FileRecord  (state=REGISTERED)            │
│    - Creates DirectoryRecord                           │
│    - Updates parent_id, root_id references             │
│    - Returns added_file_ids for Phase 2                │
│                                                         │
│ 4. DiscoveryService publishes "filesystem.updated"     │
└────────┬────────────────────────────────────────────────┘
         │
         ↓ (Auto-queue via ProcessingPipeline)
┌─────────────────────────────────────────────────────────┐
│              PHASE 2: METADATA & BASIC AI               │
├─────────────────────────────────────────────────────────┤
│ ProcessingPipeline.enqueue_phase2(file_ids)            │
│                                                         │
│ Batch size: 20 files                                   │
│                                                         │
│ Extractors (ordered by priority):                      │
│ 1. ThumbnailExtractor                                  │
│    - Generates thumbnails (multiple sizes)             │
│    - State → THUMBNAIL_READY                           │
│                                                         │
│ 2. MetadataExtractor                                   │
│    - Extracts EXIF data (camera, GPS, etc.)            │
│    - State → METADATA_READY                            │
│                                                         │
│ 3. XMPExtractor                                        │
│    - Parses XMP sidecar files                          │
│    - Extracts hierarchical tags                        │
│                                                         │
│ 4. CLIPExtractor (Optional)                            │
│    - Generates image embeddings (512-dim)              │
│    - Stores in FAISSIndexService + MongoDB             │
│    - State → INDEXED                                   │
│                                                         │
│ Result: FileRecord.processing_state = INDEXED           │
└────────┬────────────────────────────────────────────────┘
         │
         ↓ (Optional Phase 3)
┌─────────────────────────────────────────────────────────┐
│              PHASE 3: ADVANCED AI                       │
├─────────────────────────────────────────────────────────┤
│ ProcessingPipeline.enqueue_phase3(file_id)             │
│                                                         │
│ Batch size: 1 file (heavy processing)                  │
│                                                         │
│ Extractors:                                            │
│ 1. BLIPExtractor                                       │
│    - Generates image captions                          │
│    - Stores in FileRecord.ai_description               │
│                                                         │
│ 2. GroundingDINOExtractor                              │
│    - Object detection with labels                      │
│    - Creates DetectionInstance records                 │
│                                                         │
│ 3. WDTaggerExtractor                                   │
│    - Anime/illustration tagging                        │
│    - Auto-tags files under "auto/wd_tag/"              │
│                                                         │
│ 4. DetectionService (YOLO/MTCNN)                       │
│    - Face detection, object detection                  │
│    - Creates bounding boxes                            │
│                                                         │
│ Result: FileRecord.processing_state = COMPLETE          │
└────────┬────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────┐
│              SEARCH & RETRIEVAL                         │
├─────────────────────────────────────────────────────────┤
│ SearchService.search(query)                            │
│                                                         │
│ 1. MongoDB Filter Stage                                │
│    - Apply tag_ids, album_ids, file_type filters       │
│    - Fast indexed queries                              │
│                                                         │
│ 2. Vector Search Stage (if enabled)                    │
│    - FAISSIndexService.search(embedding, k=100)        │
│    - Returns similar file_ids with cosine scores       │
│                                                         │
│ 3. Merge & Rank Stage                                  │
│    - Combine filter + vector results                   │
│    - Calculate final scores                            │
│    - Sort by relevance                                 │
│                                                         │
│ 4. Return SearchResults to UI                          │
└─────────────────────────────────────────────────────────┘
```

### Processing State Machine

```python
class ProcessingState(IntEnum):
    DISCOVERED = 0       # Found on filesystem (not yet in DB)
    REGISTERED = 10      # Basic file info stored (path, size, name)
    METADATA_READY = 20  # EXIF/XMP extracted
    THUMBNAIL_READY = 30 # Thumbnails generated
    INDEXED = 40         # Embeddings stored, searchable
    ANALYZED = 50        # AI analysis complete
    COMPLETE = 100       # All processing done
```

## 3. System Interactions

### A. DiscoveryService ↔ ProcessingPipeline

```python
# In DiscoveryService._scan_root_impl()
stats = await self.sync_manager.apply_changes(diff, str(root._id))

# Auto-queue Phase 2 processing
added_ids = stats.get("added_file_ids", [])
if added_ids and self.processing_pipeline:
    await self.processing_pipeline.enqueue_phase2(added_ids)
```

**Key Points**:
- Discovery service automatically queues new files for processing
- Processes files in batches to avoid memory issues
- Uses TaskSystem for background execution

### B. ProcessingPipeline ↔ ExtractorRegistry

```python
# In ProcessingPipeline._handle_phase2_batch()
extractors = ExtractorRegistry.get_for_phase(2, locator=self.locator)

for extractor in extractors:
    processable = [f for f in files if extractor.can_process(f)]
    if processable:
        results = await extractor.process(processable)
```

**Key Points**:
- Extractors are registered at app startup
- Each extractor declares which phase it belongs to
- Extractors filtered by file type (can_process check)
- Results aggregated for progress tracking

### C. VectorService ↔ FAISSIndexService

```python
# CLIPExtractor stores embeddings
await vector_service.upsert(
    collection="clip",
    file_id=file._id,
    vector=embedding_vector  # 512-dim float array
)

# Later: SearchService retrieves
results = await self._faiss_service.search(
    collection="clip",
    query_vector=text_embedding,
    k=100,
    file_ids=filtered_ids  # Pre-filtered by MongoDB
)
```

**Key Points**:
- Embeddings stored in both FAISS (in-memory) and MongoDB (persistent)
- Multiple providers supported: clip, blip, mobilenet, thumb
- Hybrid search: MongoDB filters first, then vector similarity

### D. FileRecord as Central Hub

All systems connect through FileRecord:

```python
class FileRecord(FSRecord):
    # Discovery System
    path: str
    parent_id: ObjectId  # → DirectoryRecord
    root_id: ObjectId    # → Library Root
    
    # Processing State
    processing_state: int  # ProcessingState enum
    processing_errors: List[str]
    
    # Tag System
    tag_ids: List[ObjectId]  # → Tag references
    tags: List[str]          # Denormalized for display
    
    # Album System
    album_ids: List[ObjectId]  # → Album references
    
    # AI Results
    ai_description: str      # From BLIP
    ai_tags: List[str]       # From WDTagger
    embeddings: Dict         # CLIP, BLIP vectors
```

## 4. UExplorer Integration

### How UExplorer Uses the Pipeline

**Location**: `samples/uexplorer/main.py`

```python
# System registration order (important!)
builder = (
    ApplicationBuilder("UExplorer", str(config_path))
    .add_system(FSService)
    .add_system(MaintenanceService)
    .add_system(ProcessingPipeline)  # Must be before DiscoveryService
    .add_system(DiscoveryService)
    .add_system(ThumbnailService)
    .add_system(FAISSIndexService)   # Must be before VectorService
    .add_system(VectorService)
    .add_system(SearchService)
    # ... more services
)
```

**Why order matters**:
1. ProcessingPipeline must exist before DiscoveryService (auto-queue dependency)
2. FAISSIndexService before VectorService (dependency injection)
3. SearchService last (depends on Vector + FS services)

### UI Integration Points

#### 1. Background Tasks Panel
**File**: `samples/uexplorer/uexplorer_src/ui/docking/background_panel.py`

- Displays active tasks from TaskSystem
- Shows Phase 2/Phase 3 queue counts
- Allows manual triggering of reindex

#### 2. Metadata Panel
**File**: `samples/uexplorer/uexplorer_src/ui/widgets/metadata_panel.py`

- Shows `processing_state` indicator
- Displays which phase file is in
- Color-coded status badges

#### 3. Main Window Actions
**File**: `samples/uexplorer/uexplorer_src/ui/main_window.py`

```python
async def _reprocess_selected(self):
    """Reprocess selected files through Phase 2/3 pipeline."""
    pipeline = self.locator.get_system(ProcessingPipeline)
    task_id = await pipeline.enqueue_phase2(file_ids, force=True)
```

## 5. Key Design Patterns

### A. Plugin Architecture (Extractors)
- ExtractorRegistry uses Factory pattern
- Extractors implement common `Extractor` base class
- Easy to add new extractors without modifying pipeline

### B. Event-Driven Processing
- CommandBus publishes events: `file.created`, `file.modified`
- ProcessingPipeline subscribes and auto-queues
- UI subscribes to `processing.phase2.complete` for real-time updates

### C. Batch Processing
- Phase 1: 200 items (fast DB inserts)
- Phase 2: 20 items (moderate AI workload)
- Phase 3: 1 item (heavy AI workload)
- Prevents memory exhaustion on large collections

### D. State Machine
- ProcessingState tracks progress
- Each phase advances the state
- Allows resuming interrupted processing

## 6. Data Persistence Strategy

### MongoDB Collections Used

1. **file_records**: All files with metadata
2. **directory_records**: Directory hierarchy
3. **embeddings**: Vector embeddings (provider → file_id → vector)
4. **tags**: Hierarchical tag taxonomy
5. **albums**: User collections
6. **detections**: Object detection bounding boxes

### FAISS Indexes (In-Memory)

- Separate index per provider: `clip`, `blip`, `mobilenet`
- Loaded on startup from MongoDB `embeddings` collection
- Rebuilt automatically if missing
- Persisted to disk on shutdown

## 7. Performance Optimizations

### 1. Incremental Scanning
```python
# DiffDetector runs in incremental mode during batches
diff = await self.diff_detector.detect_changes(
    batch,
    root.path,
    incremental=True  # Skip deletion detection per batch
)

# Final pass after all batches
del_diff = await self.diff_detector.detect_deletions(
    visited_paths,  # All scanned paths
    root.path
)
```

### 2. Lazy Loading
- UI loads directory children only when expanded
- Search results paginated (limit/offset)
- Thumbnails generated on-demand

### 3. Background Processing
- Uses qasync for async/await with Qt
- TaskSystem runs workers in thread pool
- UI never blocks on heavy operations

### 4. Batch Database Operations
- SyncManager uses bulk inserts where possible
- Updates batched to reduce round-trips
- Indexes on all foreign keys

## Next Steps

1. ✅ Map complete pipeline flow
2. ✅ Identify all system interactions
3. ✅ Document UExplorer integration
4. [ ] Create comprehensive docs/indexer_pipeline.md
5. [ ] Document missing pieces (if any)
6. [ ] Update design_dock.md with findings

## Questions for User

None at this time - research is comprehensive.

## References

- [DiscoveryService](file:///d:/github/USCore/src/ucorefs/discovery/service.py)
- [ProcessingPipeline](file:///d:/github/USCore/src/ucorefs/processing/pipeline.py)
- [ExtractorRegistry](file:///d:/github/USCore/src/ucorefs/extractors/registry.py)
- [SearchService](file:///d:/github/USCore/src/ucorefs/search/service.py)
- [UExplorer Main](file:///d:/github/USCore/samples/uexplorer/main.py)

---

**Session End Time**: TBD
**Duration**: TBD
**Output**: Comprehensive documentation in docs/
