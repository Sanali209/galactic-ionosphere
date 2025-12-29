# UCore FS - Comprehensive Architecture Analysis

**Document Version**: 1.0  
**Created**: 2025-12-28  
**Research Type**: Deep Code Analysis  

## Table of Contents
1. [Overview](#overview)
2. [Current Implementation Status](#current-implementation-status)
3. [Core Components](#core-components)
4. [Missing Features & Gaps](#missing-features--gaps)
5. [Maintenance Tasks](#maintenance-tasks)

---

## Overview

**UCore FS (Universal Core Filesystem)** is a MongoDB-backed filesystem database with AI capabilities, built on the Foundation template architecture. It replaces traditional filesystem APIs with a powerful database-backed approach.

### Key Value Propositions
- **Rich Metadata**: Store and query complex metadata beyond basic file attributes
- **AI-Powered**: Semantic search using vector embeddings (CLIP, BLIP)
- **Complex Querying**: MongoDB aggregations, full-text search, vector similarity
- **Background Processing**: Three-phase pipeline (Discovery → Metadata → AI)
- **Automation**: Rules engine for workflow automation

---

## Current Implementation Status

### Implementation Phases (8 Total)

#### ✅ Phase 1: Core Schema (Complete)
**Location**: `src/ucorefs/models/`

- [FSRecord](file:///d:/github/USCore/src/ucorefs/models/base.py) - Base class for filesystem records
- [FileRecord](file:///d:/github/USCore/src/ucorefs/models/file_record.py) - File metadata model
- [DirectoryRecord](file:///d:/github/USCore/src/ucorefs/models/directory.py) - Directory hierarchy
- [FSService](file:///d:/github/USCore/src/ucorefs/services/fs_service.py) - Core filesystem operations (27KB, ~800 lines)

**Features**:
- Virtual file support
- Hierarchical organization
- Library root management
- File CRUD operations

#### ✅ Phase 2: Discovery (Complete)
**Location**: `src/ucorefs/discovery/`

- [LibraryManager](file:///d:/github/USCore/src/ucorefs/discovery/library_manager.py) - Watch/blacklist configuration
- [DirectoryScanner](file:///d:/github/USCore/src/ucorefs/discovery/scanner.py) - Batch filesystem scanning
- [DiffDetector](file:///d:/github/USCore/src/ucorefs/discovery/diff.py) - Incremental change detection
- [SyncManager](file:///d:/github/USCore/src/ucorefs/discovery/sync.py) - Atomic database updates
- [DiscoveryService](file:///d:/github/USCore/src/ucorefs/discovery/service.py) - Orchestration service

**Features**:
- Background scanning with configurable batch sizes (50 dirs, 1000 files)
- Incremental sync (only processes changes)
- Watch extensions filtering (e.g., only .jpg, .png)
- Blacklist paths support
- Thread pool offloading for disk I/O

#### ✅ Phase 3: File Types (Complete)
**Location**: `src/ucorefs/types/`

- [IFileDriver](file:///d:/github/USCore/src/ucorefs/types/driver.py) - File driver interface
- [FileTypeRegistry](file:///d:/github/USCore/src/ucorefs/types/registry.py) - Factory pattern for drivers
- Built-in drivers: Image, Text, Video, Default

**Features**:
- Polymorphic file handling
- AI methods per driver (e.g., `get_thumbnail()`, `extract_features()`)
- XMP metadata extraction with hierarchical tags

#### ✅ Phase 4: Thumbnails & Search (Complete)
**Location**: `src/ucorefs/thumbnails/`, `src/ucorefs/search/`

- [ThumbnailService](file:///d:/github/USCore/src/ucorefs/thumbnails/service.py) - Configurable thumbnail caching
- [SearchService](file:///d:/github/USCore/src/ucorefs/search/service.py) - Hybrid search (metadata + vector)

**Features**:
- On-demand thumbnail generation
- Configurable sizes
- FAISS vector index for semantic search
- Hybrid text + vector search

#### ✅ Phase 4.5: AI Pipeline (Complete)
**Location**: `src/ucorefs/ai/`, `src/ucorefs/processing/`

- [ProcessingPipeline](file:///d:/github/USCore/src/ucorefs/processing/pipeline.py) - Three-phase orchestration
- [ExtractorRegistry](file:///d:/github/USCore/src/ucorefs/extractors/registry.py) - Plugin architecture
- Extractors:
  - [CLIPExtractor](file:///d:/github/USCore/src/ucorefs/extractors/clip_extractor.py) - Vision-language embeddings
  - [BLIPExtractor](file:///d:/github/USCore/src/ucorefs/extractors/blip_extractor.py) - Image captioning
  - [GroundingDINOExtractor](file:///d:/github/USCore/src/ucorefs/extractors/grounding_dino_extractor.py) - Object detection
  - [WDTaggerExtractor](file:///d:/github/USCore/src/ucorefs/extractors/wd_tagger.py) - Waifu Diffusion tagging
  - [MetadataExtractor](file:///d:/github/USCore/src/ucorefs/extractors/metadata.py) - EXIF/XMP parsing
  - [ThumbnailExtractor](file:///d:/github/USCore/src/ucorefs/extractors/thumbnail.py) - Image resizing
- [SimilarityService](file:///d:/github/USCore/src/ucorefs/ai/similarity_service.py) - Auto-relation creation
- [LLMService](file:///d:/github/USCore/src/ucorefs/ai/llm_service.py) - Batch description generation

**Features**:
- Three-phase pipeline:
  - **Phase 1**: File discovery
  - **Phase 2**: Metadata extraction (EXIF, thumbnails, basic tags)
  - **Phase 3**: AI processing (CLIP, BLIP, object detection)
- TaskSystem integration for background processing
- Crash recovery via TaskRecord persistence
- Configurable batch sizes (PHASE2_BATCH_SIZE=20, PHASE3_BATCH_SIZE=5)

#### ✅ Phase 5: Detection & Relations (Complete)
**Location**: `src/ucorefs/detection/`, `src/ucorefs/relations/`

- [DetectionClass](file:///d:/github/USCore/src/ucorefs/detection/models.py) - Hierarchical class taxonomy (MPTT)
- [DetectionInstance](file:///d:/github/USCore/src/ucorefs/detection/models.py) - Virtual bounding boxes
- [Relation](file:///d:/github/USCore/src/ucorefs/relations/models.py) - Extensible relation system
- [RelationType](file:///d:/github/USCore/src/ucorefs/relations/models.py) - Duplicate, similar, sequence, etc.
- [DetectionService](file:///d:/github/USCore/src/ucorefs/detection/service.py) - Backend orchestration

**Features**:
- Bounding box storage as virtual files
- YOLO and MTCNN backends
- Duplicate detection (perceptual hash)
- Similar items (CLIP embeddings)
- Sequential relations (image sequences)

#### ✅ Phase 6: Tags & Albums (Complete)
**Location**: `src/ucorefs/tags/`, `src/ucorefs/albums/`

- [Tag](file:///d:/github/USCore/src/ucorefs/tags/models.py) - Hierarchical tags with MPTT
- [TagManager](file:///d:/github/USCore/src/ucorefs/tags/manager.py) - Tag CRUD and hierarchy management
- [Album](file:///d:/github/USCore/src/ucorefs/albums/models.py) - Manual and smart albums
- [AlbumManager](file:///d:/github/USCore/src/ucorefs/albums/manager.py) - Album management

**Features**:
- Hierarchical tags (e.g., `Animals/Mammals/Cat`)
- Tag synonyms/antonyms for search expansion
- Manual albums (static file lists)
- Smart albums (dynamic MongoDB queries)
- Tag full_path denormalization for display
- File count tracking

**See**: [Tag, Directory, and Album Systems Architecture](file:///d:/github/USCore/docs/tag_directory_album_systems.md) (798 lines)

#### ✅ Phase 7: Rules Engine (Complete)
**Location**: `src/ucorefs/rules/`

- [Rule](file:///d:/github/USCore/src/ucorefs/rules/models.py) - Rule model with triggers
- [RulesEngine](file:///d:/github/USCore/src/ucorefs/rules/engine.py) - Rule execution engine
- [ICondition](file:///d:/github/USCore/src/ucorefs/rules/conditions.py) - Extensible condition interface
- [IAction](file:///d:/github/USCore/src/ucorefs/rules/actions.py) - Extensible action interface

**Features**:
- Triggers: `on_import`, `on_tag`, `manual`
- Built-in conditions: path_contains, extension_in, has_tag, etc.
- Built-in actions: add_tag, move_file, set_rating, etc.
- Plugin architecture for custom conditions/actions

#### ✅ Phase 8: Query Builder (Complete)
**Location**: `src/ucorefs/query/`

- [QueryBuilder](file:///d:/github/USCore/src/ucorefs/query/builder.py) - Fluent API for queries
- [Q](file:///d:/github/USCore/src/ucorefs/query/builder.py) - Query condition helpers
- [Aggregation](file:///d:/github/USCore/src/ucorefs/query/aggregations.py) - MongoDB aggregation pipelines

**Features**:
- Fluent API: `QueryBuilder().AND(...).NOT(...).order_by(...).limit(...)`
- `Q` expressions: `Q.rating_gte(4)`, `Q.has_tag(tag_id)`, `Q.OR(...)`
- Vector search integration
- Aggregation helpers (group_by_tag, group_by_album, etc.)

---

## Core Components

### 1. FSService (Filesystem Operations)
**Location**: [src/ucorefs/services/fs_service.py](file:///d:/github/USCore/src/ucorefs/services/fs_service.py)  
**Size**: 27KB (~800 lines)

**Responsibilities**:
- Library root management (`add_library_root()`, `get_roots()`)
- File/directory CRUD (`create_file()`, `upsert_file()`, `delete_file()`)
- Hierarchy navigation (`get_children()`, `get_files()`, `get_directories()`)
- Path-based lookup (`get_by_path()`)
- File operations (`move_file()`, `copy_file()`, `rename_file()`)
- Name-based search (`search_by_name()`)

**Integration**:
- Uses `DirectoryRecord` and `FileRecord` models
- Publishes events via `EventBus` (`file.created`, `file.modified`, `file.deleted`)
- Registers with `ServiceLocator`

### 2. DiscoveryService (Background Scanning)
**Location**: [src/ucorefs/discovery/service.py](file:///d:/github/USCore/src/ucorefs/discovery/service.py)

**Responsibilities**:
- Orchestrate directory scanning
- Detect file changes (added, modified, deleted)
- Sync database with filesystem state
- Queue files for Phase 2 processing

**Components**:
- **LibraryManager**: Manages library roots, watch extensions, blacklists
- **DirectoryScanner**: Walks filesystem using `os.scandir()`, yields batches
- **DiffDetector**: Compares scan results with database state
- **SyncManager**: Applies changes atomically

**Performance**:
- Uses dedicated thread pool for disk I/O (prevents UI blocking)
- Batch sizes: 50 directories, 1000 files
- Incremental processing

### 3. ProcessingPipeline (AI Indexing)
**Location**: [src/ucorefs/processing/pipeline.py](file:///d:/github/USCore/src/ucorefs/processing/pipeline.py)

**Three-Phase Architecture**:

```
Phase 1: Discovery
    ↓ (DirectoryScanner)
Phase 2: Metadata Extraction (Batch=20)
    - EXIF/XMP parsing
    - Thumbnail generation
    - Basic file info
    ↓
Phase 3: AI Processing (Batch=5)
    - CLIP embeddings
    - BLIP captions
    - Object detection
    - WD Tagger
```

**State Machine**:
- `DISCOVERED` → `PHASE2_PENDING` → `PHASE2_PROCESSING` → `PHASE2_COMPLETE`
- `PHASE2_COMPLETE` → `PHASE3_PENDING` → `PHASE3_PROCESSING` → `INDEXED`

**See**: [Indexer Pipeline Architecture](file:///d:/github/USCore/docs/indexer_pipeline_architecture.md) for detailed flow diagrams

### 4. ExtractorRegistry (Plugin System)
**Location**: [src/ucorefs/extractors/registry.py](file:///d:/github/USCore/src/ucorefs/extractors/registry.py)

**Registered Extractors**:

| Extractor | Phase | Description |
|-----------|-------|-------------|
| MetadataExtractor | 2 | EXIF/XMP parsing with pyexiv2 |
| ThumbnailExtractor | 2 | PIL-based thumbnail generation |
| CLIPExtractor | 3 | Vision-language embeddings (OpenAI CLIP) |
| BLIPExtractor | 3 | Image captioning (Salesforce BLIP) |
| GroundingDINOExtractor | 3 | Object detection with grounding |
| WDTaggerExtractor | 3 | Waifu Diffusion tag prediction |

**Extractor Interface**:
```python
class IExtractor:
    phase: int  # 2 or 3
    is_cpu_heavy: bool  # For thread pool offloading
    
    async def process(self, files: List[FileRecord]) -> Dict:
        """Process batch of files, return results"""
```

### 5. VectorService (Semantic Search)
**Location**: [src/ucorefs/vectors/service.py](file:///d:/github/USCore/src/ucorefs/vectors/service.py)

**Features**:
- FAISS index for vector embeddings
- Store embeddings from CLIP, BLIP
- Similarity search with cosine distance
- Hybrid search (text + vector)

**Vector Types**:
- `clip_embeddings` (512-dim or 768-dim)
- `blip_embeddings` (768-dim)

### 6. SearchService (Unified Search)
**Location**: [src/ucorefs/search/service.py](file:///d:/github/USCore/src/ucorefs/search/service.py)

**Search Types**:
1. **Metadata Search**: MongoDB queries (fast, exact matches)
2. **Full-Text Search**: MongoDB text indexes (keyword search)
3. **Vector Search**: FAISS similarity (semantic search)
4. **Hybrid Search**: Combined scoring (metadata + vector)

**SearchQuery Model**:
```python
class SearchQuery:
    text: Optional[str]  # Text query
    filters: Dict  # MongoDB filters (tag_ids, rating, etc.)
    vector_search: bool  # Enable semantic search
    limit: int
    offset: int
```

### 7. MaintenanceService (Database Cleanup)
**Location**: [src/ucorefs/services/maintenance_service.py](file:///d:/github/USCore/src/ucorefs/services/maintenance_service.py)  
**Size**: 14KB (~400 lines)

**Features** (From Phase 6 implementation):
- **Remove Missing Files**: Delete records for files that no longer exist on disk
- **Remove Orphaned Hashes**: Clean up hash table entries without corresponding files
- **Remove Orphaned Vector Entries**: Clean FAISS index entries
- **Vacuum Database**: Optimize MongoDB collections
- **Generate Report**: Statistics on cleanup operations

---

## Missing Features & Gaps

### Critical Missing Features

#### 1. **Annotation System** ❌ NOT IMPLEMENTED
**Status**: Mentioned in roadmap ("resarch anotation system"), but no implementation found  
**Location**: `src/ucorefs/annotation/` exists but may be incomplete  

**Required Features**:
- AnnotationRecord model (bounding boxes, polygons, keypoints)
- Annotation types: rectangle, polygon, polyline, point
- Annotation classes/labels
- User attribution (who created annotation)
- Versioning (annotation history)
- Export formats (COCO, YOLO, Pascal VOC)

**Implementation Plan Needed**:
```python
# Proposed model
class Annotation(CollectionRecord):
    file_id: ObjectId
    annotation_type: str  # rectangle, polygon, keypoint
    geometry: Dict  # {x, y, width, height} or {points: [...]}
    class_id: Optional[ObjectId]
    label: str
    confidence: float  # For AI-generated annotations
    created_by: str  # User or AI model name
```

#### 2. **Reference System** ❌ NOT IMPLEMENTED
**Status**: Mentioned in roadmap ("resarch reference system"), no implementation  

**Potential Use Cases**:
- Cross-file references (e.g., "source material" for derivatives)
- Citation tracking
- Dependency graphs (e.g., PSD → exported PNG)
- Version chains (v1 → v2 → v3)

**Could Leverage Existing**:
- Relation system (already has `parent_id`, `child_id`, `relation_type`)
- Extend `RelationType` enum with reference-specific types

#### 3. **Viewport Priority Queue** ⚠️ PARTIALLY IMPLEMENTED
**Status**: Priority queue implemented (Phase 3 of SAN-14), but UI integration deferred  
**See**: [TaskSystem SAN-14 Optimization](file:///d:/github/USCore/docs/tasksystem_san14_optimization.md)

**What Exists**:
- `TaskSystem` supports priority levels (0=HIGH, 1=NORMAL, 2=LOW)
- `ProcessingPipeline.submit()` accepts priority parameter

**What's Missing**:
- UI integration to detect visible files in viewport
- Auto-queue visible files with HIGH priority
- FilePaneWidget doesn't signal viewport changes

**Recommended Implementation** (from research):
```python
# In FilePaneWidget (UExplorer)
def _on_viewport_changed(self):
    """Called when user scrolls or changes view"""
    visible_file_ids = self._get_visible_files()
    
    # Queue visible files with high priority
    for file_id in visible_file_ids:
        if file.processing_state == ProcessingState.PHASE2_PENDING:
            await pipeline.enqueue_phase2([file_id], priority=0)  # HIGH
```

#### 4. **Configuration Tuning for Performance** ⚠️ PARTIAL
**Status**: Manual configuration exists, but no auto-detection  
**Proposed**: [Roadmap Phase 5](file:///d:/github/USCore/docs/roadmap.md#L98-L101)

**What Exists**:
- Config file (`config.json`) with manual settings
- Batch size configs: `PHASE2_BATCH_SIZE`, `PHASE3_BATCH_SIZE`
- Worker count: `config.general.task_workers`

**What's Missing**:
- System capability detection (GPU type, RAM, CPU cores)
- Auto-tuning profiles (Low-end, Medium, High-end, Server)
- Performance benchmarking tools
- Configuration validation with warnings

#### 5. **Checkpoint/Resume Pipeline State** ❌ NOT IMPLEMENTED
**Proposed**: [Roadmap Phase 5](file:///d:/github/USCore/docs/roadmap.md#L83-L86)

**Problem**: If app crashes during pipeline processing:
- `_phase2_pending` and `_phase3_pending` in-memory sets are lost
- Files stuck in `PROCESSING` state forever
- User must manually trigger re-scan

**Proposed Solution**:
- Serialize pending sets to MongoDB on periodic checkpoints
- On startup, restore pending sets and resume
- Add "Clear Pipeline State" admin action

### Non-Critical Gaps

#### 6. **System Bundles** (SAN-13) ⚠️ DEFERRED
**Status**: Ticket exists, implementation deferred  
**Problem**: `main.py` has verbose service registration  
**See**: [Roadmap unsorted](file:///d:/github/USCore/docs/roadmap.md#L112-L114)

```python
# Current: main.py
app.register_system(FSService)
app.register_system(DiscoveryService)
app.register_system(ProcessingPipeline)
# ... 10+ more

# Proposed: SystemBundle
class UCoreFSBundle(SystemBundle):
    def register(self, builder):
        builder.register_system(FSService)
        builder.register_system(DiscoveryService)
        # ...

# Usage
app.register_bundle(UCoreFSBundle)
```

#### 7. **Loading Dialog** (SAN-8) ❌ NOT IMPLEMENTED
**Status**: Ticket exists, no implementation  
**See**: [Roadmap unsorted](file:///d:/github/USCore/docs/roadmap.md#L104-L110)

**Requirement**: Modal loading indicator for long-running operations  
**Implementation**: Create `LoadingDialog` class in `src.ui.dialogs`

#### 8. **Docking Auto-Hide** (SAN-23) ❌ NOT IMPLEMENTED
**Status**: Todo in roadmap  
**See**: [Roadmap Phase 3](file:///d:/github/USCore/docs/roadmap.md#L61-L63)

**Feature**: Implement "unpin" functionality for dock panels (like VS Code)

---

## Maintenance Tasks

### High Priority

#### 1. **[SAN-31] Remove Obsolete ChromaDB References**
**Status**: Todo (High Priority)  
**Context**: Remove obsolete references from `src/ucorefs/README.md`. Code is already clean.  
**Location**: [Roadmap Phase 1](file:///d:/github/USCore/docs/roadmap.md#L8-L10)

**Action**: Update README to reflect FAISS-only vector storage.

#### 2. **[SAN-32] Update Legacy Documentation Paths**
**Status**: Backlog  
**Context**: Fix strict path references in `docs/` to match `src/` structure  
**Location**: [Roadmap Phase 1](file:///d:/github/USCore/docs/roadmap.md#L11-L13)

**Affected Files** (likely):
- `docs/architecture.md`
- `docs/modules.md`
- `docs/tutorials.md`

#### 3. **[SAN-19] USCore Architectural Audit & Refactor**
**Status**: Todo  
**Location**: [Roadmap Phase 2](file:///d:/github/USCore/docs/roadmap.md#L19-L24)

**Tasks**:
- **Events**: Deprecate `ObserverEvent` (sync) in favor of `EventBus` (unified)
- **ORM**: Fix inheritance gaps and enforce rigid `DatabaseManager` usage
- **Logging**: Standardize logging/config

**Impact**: Foundation-level changes affecting all systems

### Medium Priority

#### 4. **Research Annotation System**
**Status**: Not started  
**Requirement**: Design and implement annotation system for bounding boxes, polygons, etc.

**Deliverables**:
- Research document analyzing requirements
- Model design (AnnotationRecord, AnnotationClass)
- Integration plan with DetectionService
- Export format support

#### 5. **Research Reference System**
**Status**: Not started  
**Requirement**: Design cross-file reference system

**Deliverables**:
- Research document on use cases
- Evaluation of existing Relation system for extension
- Model design if separate system needed

#### 6. **Implement Viewport Priority Queue UI Integration**
**Status**: Backend complete, UI integration deferred  
**See**: [SAN-14 Complete](file:///d:/github/USCore/docs/roadmap.md#L26-L56)

**Required**:
- Modify `FilePaneWidget` in UExplorer to detect visible files
- Signal viewport changes to `ProcessingPipeline`
- Queue visible files with `priority=0` (HIGH)

### Low Priority

#### 7. **Implement Configuration Tuning**
**Status**: Proposed (Roadmap Phase 5)  
**Requirement**: Auto-detect system capabilities and suggest optimal config

#### 8. **Implement Checkpoint/Resume**
**Status**: Proposed (Roadmap Phase 5)  
**Requirement**: Persist pipeline state for crash recovery

#### 9. **Create System Bundles (SAN-13)**
**Status**: Deferred  
**Requirement**: Refactor `main.py` to use `SystemBundle` pattern

#### 10. **Add Loading Dialog (SAN-8)**
**Status**: Todo  
**Requirement**: Create `LoadingDialog` for long-running operations

---

## Architecture Strengths

### What's Working Well

1. **Modular Design**: Clear separation of concerns across 18 subsystems
2. **Foundation Integration**: Proper use of ServiceLocator, BaseSystem, EventBus
3. **Background Processing**: TaskSystem with crash recovery and persistence
4. **Extensibility**: Plugin architectures (Extractors, FileDrivers, Rules)
5. **Performance**: Batch processing, thread pool offloading, lazy loading
6. **Testing**: 74 tests across 8 phases
7. **Documentation**: Comprehensive docs for major systems

### Proven Patterns

- **MPTT Trees**: Used for Tag hierarchy, DetectionClass hierarchy (efficient subtree queries)
- **Denormalization**: Tag full_paths stored as strings for display (trade storage for speed)
- **Lazy Imports**: Avoid circular dependencies in `__init__.py`
- **State Machine**: ProcessingState enum for pipeline orchestration
- **Repository Pattern**: FSService, TagManager, AlbumManager abstract database access
- **Observer Pattern**: EventBus for decoupled system notifications

---

## Dependencies

### Core
- `motor` - Async MongoDB driver
- `beanie` - Async ORM (Pydantic + motor)
- `pydantic` - Data validation
- `loguru` - Structured logging

### AI & ML
- `torch` - PyTorch for AI models
- `transformers` - Hugging Face models (CLIP, BLIP)
- `Pillow` (PIL) - Image processing
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `imagehash` - Perceptual hashing for duplicates

### Optional
- `pyexiv2` - XMP metadata extraction (C++, releases GIL)
- `chromadb` - **OBSOLETE** (replaced by FAISS)

---

## Testing Coverage

### Test Distribution
- **Total Tests**: 74 tests
- **Implementation**: ~7,200 lines of code
- **Test Ratio**: ~1:97 (could be improved)

### Test Organization
```
tests/ucorefs/
├── test_phase1_core.py          # FSService, models
├── test_phase2_discovery.py     # Scanner, diff, sync
├── test_phase3_types.py         # FileDrivers, registry
├── test_phase4_search.py        # SearchService, ThumbnailService
├── test_phase5_detection.py     # DetectionService, relations
├── test_phase6_tags_albums.py   # TagManager, AlbumManager
├── test_phase7_rules.py         # RulesEngine, conditions, actions
└── test_phase8_query.py         # QueryBuilder, aggregations
```

**Recent Additions** (2025-12-28):
- EventBus: 100% coverage (400+ lines, 25+ tests)
- FSService: 80% coverage (500+ lines, 30+ tests)

**See**: [Phase 4 Unit Tests Summary](file:///d:/github/USCore/dev_log/phase4_unit_tests_summary.md)

---

## Version History

- **v0.2.0**: Current version (from `__init__.py`)
- Phase 8 complete (Query Builder)
- SAN-14 optimization complete (TaskSystem improvements)

---

## References

### Documentation
- [UCore FS README](file:///d:/github/USCore/src/ucorefs/README.md)
- [Tag/Directory/Album Systems](file:///d:/github/USCore/docs/tag_directory_album_systems.md)
- [Indexer Pipeline Architecture](file:///d:/github/USCore/docs/indexer_pipeline_architecture.md)
- [TaskSystem SAN-14 Optimization](file:///d:/github/USCore/docs/tasksystem_san14_optimization.md)
- [Roadmap](file:///d:/github/USCore/docs/roadmap.md)

### Session Journals
- [TaskSystem Research](file:///d:/github/USCore/dev_log/journal_session_tasksystem_research.md)
- [Indexer Pipeline Research](file:///d:/github/USCore/dev_log/journal_session_indexer_pipeline_research.md)
- [Tag/Dir/Album Research](file:///d:/github/USCore/dev_log/journal_session_tag_dir_album_research.md)

### Source Code
- **Models**: `src/ucorefs/models/`
- **Services**: `src/ucorefs/services/`
- **Discovery**: `src/ucorefs/discovery/`
- **Processing**: `src/ucorefs/processing/`
- **Extractors**: `src/ucorefs/extractors/`
