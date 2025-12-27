# Tag, Directory, and Album Systems Architecture

**Document Version**: 1.0  
**Last Updated**: 2025-12-27  
**Research Depth**: Deep Code Analysis

## Table of Contents
1. [Overview](#overview)
2. [Tag System](#tag-system)
3. [Directory System](#directory-system)
4. [Album System](#album-system)
5. [Integration & Data Flow](#integration--data-flow)
6. [File Count Implementation](#file-count-implementation)
7. [UI Layer](#ui-layer)
8. [Search & Filtering](#search--filtering)

---

## Overview

USCore implements three organizational systems for managing files:

- **Tag System**: Hierarchical taxonomy for categorizing files (e.g., `Animals/Mammals/Cats`)
- **Directory System**: Filesystem-based hierarchical organization with library roots
- **Album System**: User-curated collections (manual or smart/dynamic)

All three systems work together through **ObjectId references** stored on `FileRecord` instances.

### Core Principles

1. **Hierarchical Organization**: All three systems support parent-child relationships
2. **File Count Tracking**: Each system tracks how many files are associated
3. **Include/Exclude Filtering**: UI panels allow combined filtering
4. **Denormalization**: Critical data is stored redundantly for performance (e.g., tag names as strings)

---

## Tag System

### Architecture

**Location**: `src/ucorefs/tags/`

```
src/ucorefs/tags/
├── models.py        # Tag model with MPTT structure
└── manager.py       # TagManager service
```

### Data Model

**File**: [models.py](file:///d:/github/USCore/src/ucorefs/tags/models.py)

```python
class Tag(CollectionRecord):
    # Tag info
    name: str = Field(default="", index=True)
    full_path: str = Field(default="", index=True)
    
    # Hierarchy (MPTT - Modified Preorder Tree Traversal)
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    lft: int = Field(default=0, index=True)
    rgt: int = Field(default=0, index=True)
    depth: int = Field(default=0)
    
    # Statistics
    file_count: int = Field(default=0)
    
    # Relationships
    synonym_ids: List[ObjectId] = Field(default_factory=list)
    antonym_ids: List[ObjectId] = Field(default_factory=list)
    
    # UI
    color: str = Field(default="")
```

**Key Features**:
- **MPTT Structure**: Enables efficient subtree queries using `lft` and `rgt` values
- **Full Path**: Stored for display (e.g., "Animals/Mammals/Cat")
- **File Count**: Cached count updated when files are tagged
- **Synonyms/Antonyms**: Bidirectional relationships for search expansion

### Tag Manager

**File**: [manager.py](file:///d:/github/USCore/src/ucorefs/tags/manager.py#L15-L572)

**Core Methods**:

```python
class TagManager(BaseSystem):
    async def create_tag(name: str, parent_id: Optional[ObjectId], color: str) -> Tag
    async def create_tag_from_path(tag_path: str, delimiter: str) -> Tag  # Parses "a/b/c"
    async def add_tag_to_file(file_id: ObjectId, tag_path: str) -> bool
    
    # Relationships
    async def add_synonym(tag_id: ObjectId, synonym_id: ObjectId) -> bool
    async def add_antonym(tag_id: ObjectId, antonym_id: ObjectId) -> bool
    async def expand_search_with_synonyms(tag_ids: List[ObjectId]) -> Set[ObjectId]
    
    # Hierarchy
    async def get_children(tag_id: Optional[ObjectId]) -> List[Tag]
    async def delete_tag(tag_id: ObjectId, recursive: bool) -> bool
    
    # Statistics
    async def get_tag_statistics() -> dict
    async def bulk_rename(old_prefix: str, new_prefix: str) -> dict
```

### Tag Creation Flow

```
1. User creates tag "Animals/Mammals/Cat"
   ↓
2. TagManager.create_tag_from_path() parses path
   ↓
3. Creates/gets "Animals" (root, depth=0)
   ↓
4. Creates/gets "Mammals" (parent=Animals, depth=1)
   ↓
5. Creates "Cat" (parent=Mammals, depth=2)
   ↓
6. Returns leaf tag "Cat" with full_path="Animals/Mammals/Cat"
```

### File Tagging Flow

```
1. User drops file onto tag in UI
   ↓
2. TagManager.add_tag_to_file(file_id, "auto/wd_tag/sky")
   ↓
3. Ensures tag exists (creates if needed)
   ↓
4. Updates FileRecord:
      file.tag_ids.append(tag._id)           # ObjectId reference
      file.tags.append("auto/wd_tag/sky")    # Denormalized string
   ↓
5. Increments Tag.file_count
   ↓
6. Publishes "file.modified" event (for UI refresh)
```

---

## Directory System

### Architecture

**Location**: `src/ucorefs/models/` and `src/ucorefs/services/`

```
src/ucorefs/
├── models/
│   ├── base.py               # FSRecord base class
│   ├── directory.py          # DirectoryRecord model
│   └── file_record.py        # FileRecord model
├── services/
│   └── fs_service.py         # Filesystem operations
└── discovery/
    └── scanner.py            # DirectoryScanner
```

### Data Model

**File**: [directory.py](file:///d:/github/USCore/src/ucorefs/models/directory.py)

```python
class DirectoryRecord(FSRecord):
    # Directory statistics
    child_count: int = Field(default=0)
    file_count: int = Field(default=0)  # Recursive count
    total_size: int = Field(default=0)  # Total size of contents
    
    # Library root settings
    is_root: bool = Field(default=False, index=True)
    
    # Watch/scan settings (for library roots)
    watch_extensions: list = Field(default_factory=list)
    blacklist_paths: list = Field(default_factory=list)
    scan_enabled: bool = Field(default=True)
```

**FSRecord Base** (inherited):
```python
class FSRecord(CollectionRecord):
    path: str = Field(default="", index=True)        # Full path
    name: str = Field(default="", index=True)        # Filename/dirname
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    root_id: Optional[ObjectId] = Field(default=None, index=True)
    size_bytes: int = Field(default=0)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### FSService

**File**: [fs_service.py](file:///d:/github/USCore/src/ucorefs/services/fs_service.py#L18-L624)

**Core Methods**:

```python
class FSService(BaseSystem):
    # Entry Points
    async def get_roots() -> List[DirectoryRecord]
    async def get_children(dir_id: ObjectId, limit: int, skip: int) -> List[FSRecord]
    async def get_files(dir_id: ObjectId) -> List[FileRecord]
    async def get_directories(dir_id: ObjectId) -> List[DirectoryRecord]
    async def get_by_path(path: str) -> Optional[FSRecord]
    async def search_by_name(pattern: str, file_type: str, limit: int) -> List[FSRecord]
    
    # CRUD
    async def create_file(path: str, name: str, parent_id: ObjectId, **kwargs) -> FileRecord
    async def upsert_file(...) -> FileRecord
    async def create_directory(...) -> DirectoryRecord
    async def upsert_directory(...) -> DirectoryRecord
    async def add_library_root(path: str, watch_extensions: List[str], blacklist_paths: List[str]) -> DirectoryRecord
    
    # File Operations
    async def move_file(file_id: ObjectId, new_folder: str, conflict_resolution: str) -> dict
    async def copy_file(file_id: ObjectId, dest_folder: str, conflict_resolution: str) -> dict
    async def rename_file(file_id: ObjectId, new_name: str) -> dict
```

### Directory Hierarchy

```
DirectoryRecord (is_root=True, id=ROOT_ID)
├── path: "D:/Photos"
├── root_id: ROOT_ID (self-reference)
├── watch_extensions: ["jpg", "png"]
└── DirectoryRecord (parent_id=ROOT_ID)
    ├── path: "D:/Photos/2023"
    ├── root_id: ROOT_ID
    └── FileRecord (parent_id=2023_ID)
        ├── path: "D:/Photos/2023/IMG_001.jpg"
        ├── root_id: ROOT_ID
        ├── parent_id: 2023_ID
        ├── tag_ids: [tag1_id, tag2_id]
        └── album_ids: [album1_id]
```

### Directory Scanning

**File**: [scanner.py](file:///d:/github/USCore/src/ucorefs/discovery/scanner.py)

```python
class DirectoryScanner:
    def scan_directory(root_path: str, watch_extensions: List[str], 
                       blacklist_paths: List[str], recursive: bool) -> Iterator[List[ScanResult]]:
        """
        Yields batches of ScanResult:
        - Directories in batches of 50 (for fast tree population)
        - Files in batches of 1000 (for performance)
        """
```

**Scan Flow**:
```
1. FSService.add_library_root("D:/Photos", watch_extensions=["jpg"])
   ↓
2. DirectoryScanner.scan_directory_async() runs in thread pool
   ↓
3. os.scandir() walks filesystem
   ↓
4. Yields DirectoryRecord batches (50 at a time)
   ↓
5. Yields FileRecord batches (1000 at a time)
   ↓
6. Database writes happen incrementally
   ↓
7. UI updates incrementally (prevents freezing)
```

---

## Album System

### Architecture

**Location**: `src/ucorefs/albums/`

```
src/ucorefs/albums/
├── models.py        # Album model
└── manager.py       # AlbumManager service
```

### Data Model

**File**: [models.py](file:///d:/github/USCore/src/ucorefs/albums/models.py)

```python
class Album(CollectionRecord):
    # Album info
    name: str = Field(default="", index=True)
    description: str = Field(default="")
    
    # Hierarchy
    parent_id: Optional[ObjectId] = Field(default=None, index=True)
    
    # Cover image
    cover_id: Optional[ObjectId] = Field(default=None)
    
    # Smart album (dynamic query-based)
    is_smart: bool = Field(default=False, index=True)
    smart_query: Dict[str, Any] = Field(default_factory=dict)  # MongoDB query
    
    # Manual album (static file list)
    file_ids: list = Field(default_factory=list)
    
    # Statistics
    file_count: int = Field(default=0)
```

### Album Types

#### 1. Manual Album (Static)
```python
Album(
    name="Vacation 2023",
    is_smart=False,
    file_ids=[file1_id, file2_id, file3_id],
    file_count=3
)
```

#### 2. Smart Album (Dynamic)
```python
Album(
    name="Recent High-Rated Images",
    is_smart=True,
    smart_query={
        "file_type": "image",
        "rating": {"$gte": 4},
        "created_at": {"$gte": last_month}
    },
    file_count=0  # Calculated dynamically
)
```

### AlbumManager

**File**: [manager.py](file:///d:/github/USCore/src/ucorefs/albums/manager.py#L15-L226)

```python
class AlbumManager(BaseSystem):
    async def create_album(name: str, description: str, parent_id: ObjectId,
                           is_smart: bool, smart_query: Dict) -> Album
    
    # Manual albums
    async def add_file_to_album(album_id: ObjectId, file_id: ObjectId) -> bool
    async def remove_file_from_album(album_id: ObjectId, file_id: ObjectId) -> bool
    
    # Get files
    async def get_album_files(album_id: ObjectId, limit: int, offset: int) -> List[FileRecord]
    
    # Smart albums
    async def update_smart_query(album_id: ObjectId, query: Dict) -> bool
    async def _execute_smart_query(query: Dict, limit: int, offset: int) -> List[FileRecord]
```

### Smart Album Query Execution

```
1. User selects smart album "High Rated"
   ↓
2. UI requests AlbumManager.get_album_files(album_id)
   ↓
3. AlbumManager detects is_smart=True
   ↓
4. Executes smart_query against FileRecord collection
      smart_query = {"rating": {"$gte": 4}}
   ↓
5. Returns dynamically filtered results
   ↓
6. UI displays files (file_count calculated on-the-fly)
```

---

## Integration & Data Flow

### FileRecord as Central Hub

```python
class FileRecord(FSRecord):
    # File type info
    file_type: str = Field(default="unknown", index=True)
    extension: str = Field(default="", index=True)
    hash_md5: str = Field(default="", index=True)
    
    # Tags (both normalized and denormalized)
    tag_ids: List[ObjectId] = Field(default_factory=list)      # ← Tag system integration
    tags: List[str] = Field(default_factory=list)              # ← Denormalized for display
    
    # Albums
    album_ids: List[ObjectId] = Field(default_factory=list)    # ← Album system integration
    
    # Directory hierarchy (inherited from FSRecord)
    parent_id: Optional[ObjectId]  # ← Directory system integration
    root_id: Optional[ObjectId]
```

### Complete Data Flow

```
┌─────────────────┐
│  Filesystem     │
│  (OS)           │
└────────┬────────┘
         │
         ↓ (DirectoryScanner)
┌─────────────────┐         ┌──────────────┐
│ DirectoryRecord │ ←───────│  FSService   │
│ (is_root=True)  │         └──────────────┘
└────────┬────────┘
         │ parent_id
         ↓
┌─────────────────┐
│ DirectoryRecord │
│ (subdirectory)  │
└────────┬────────┘
         │ parent_id
         ↓
┌─────────────────────────────────────────────┐
│            FileRecord                       │
│  ┌──────────────────────────────────────┐  │
│  │ tag_ids: [ObjectId, ...]             │──┼──→ Tag (name, file_count)
│  │ tags: ["auto/sky", "manual/nature"]  │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │ album_ids: [ObjectId, ...]           │──┼──→ Album (is_smart, file_ids)
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │ parent_id: ObjectId                  │──┼──→ DirectoryRecord
│  │ root_id: ObjectId                    │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
         │
         ↓ (SearchService, UI)
┌─────────────────┐
│   User Query    │
│  + Tag Filter   │
│  + Album Filter │
│  + Path Filter  │
└─────────────────┘
```

---

## File Count Implementation

### Tag File Count

**How counts are tracked**:

1. **Eager Update** (when file is tagged):
```python
# In TagManager.add_tag_to_file()
tag.file_count = tag.file_count + 1
await tag.save()
```

2. **Display in UI**:
```python
# In TagTreeWidget
if tag.file_count > 0:
    item.setText(0, f"{tag.name} ({tag.file_count})")
```

3. **Calculation via Aggregation**:
```python
# Alternative: Real-time count via aggregation
pipeline = [
    {"$unwind": "$tag_ids"},
    {"$group": {
        "_id": "$tag_ids",
        "count": {"$sum": 1}
    }}
]
```

### Directory File Count

```python
class DirectoryRecord:
    child_count: int = 0      # Number of immediate children (files + dirs)
    file_count: int = 0       # Total files recursively
    total_size: int = 0       # Sum of all file sizes
```

**Calculation** (during scan):
```python
# After scanning directory  
child_count = len(immediate_files) + len(immediate_subdirs)
file_count = len(all_files_recursively)
total_size = sum(file.size_bytes for file in all_files_recursively)
```

### Album File Count

**Manual Album**:
```python
album.file_count = len(album.file_ids)
```

**Smart Album**:
```python
# Calculated on-the-fly
file_count = await FileRecord.count_documents(album.smart_query)
```

---

## UI Layer

### Panel Architecture

All three systems have dedicated UI panels in UExplorer:

```
samples/uexplorer/uexplorer_src/ui/docking/
├── tag_panel.py          # Tag navigation + filtering
├── album_panel.py        # Album navigation + filtering
└── directory_panel.py    # Directory tree navigation
```

### TagPanel

**File**: [tag_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/tag_panel.py)

```python
class TagPanel(PanelBase):
    filter_changed = Signal(list, list)  # (include_tag_ids, exclude_tag_ids)
    
    def __init__(self, parent, locator):
        self._include_tags: set = set()
        self._exclude_tags: set = set()
        self._tree = TagTreeWidget(locator)
    
    def toggle_include(self, tag_id: str):
        """Add/remove tag from include filter"""
        if tag_id in self._include_tags:
            self._include_tags.discard(tag_id)
        else:
            self._include_tags.add(tag_id)
            self._exclude_tags.discard(tag_id)  # Can't be both
        self.filter_changed.emit(list(self._include_tags), list(self._exclude_tags))
```

**Features**:
- Right-click: Include (I) / Exclude (E)
- Drag-drop files onto tags to apply tags
- Displays tag hierarchy with file counts
- Clear filters button

### AlbumPanel

**File**: [album_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/album_panel.py)

```python
class AlbumPanel(PanelBase):
    filter_changed = Signal(list, list)  # (include_album_ids, exclude_album_ids)
    smart_album_selected = Signal(str, dict)  # album_id, query_dict
    
    def __init__(self, parent, locator):
        self._album_manager = locator.get_system(AlbumManager)
        self._include_albums: set = set()
        self._exclude_albums: set = set()
```

**Features**:
- Create new albums with "+" button
- Include/exclude filtering (like tags)
- Smart album creation dialog

### DirectoryPanel

**File**: [directory_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/directory_panel.py)

```python
class DirectoryPanel(PanelBase):
    directory_selected = Signal(str, str)  # directory_id, path
    filter_changed = Signal(list, list)    # (include_paths, exclude_paths)
    
    async def _load_roots(self):
        """Load library roots as top-level items"""
        roots = await self._fs_service.get_roots()
        for root in roots:
            item = self._create_dir_item(root)
            # Add placeholder for lazy loading
    
    async def _load_children(self, parent_item, dir_id):
        """Lazy load subdirectories when expanded"""
        subdirs = await self._fs_service.get_directories(ObjectId(dir_id))
```

**Features**:
- Lazy loading (only loads children when expanded)
- Refresh button
- Include/exclude directory filtering
- Shows file/folder icons

### TagTreeWidget

**File**: [tag_tree.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/widgets/tag_tree.py)

**Drag-and-Drop Tagging**:

```python
class TagTreeWidget(QTreeWidget):
    def dropEvent(self, event: QDropEvent):
        """Handle files dropped on tag"""
        tag_id = item.data(0, Qt.UserRole)
        file_ids = event.mimeData().data('application/x-file-ids')
        
        # Apply tags async
        asyncio.ensure_future(self._apply_tag_to_files(tag_id, file_ids))
    
    async def _apply_tag_to_files(self, tag_id: str, file_ids: list):
        """Apply tag to dropped files"""
        for file_id in file_ids:
            file_record = await FileRecord.get(ObjectId(file_id))
            if ObjectId(tag_id) not in file_record.tag_ids:
                file_record.tag_ids.append(ObjectId(tag_id))
                await file_record.save()
        
        # Update tag file count
        tag.file_count += count
        await tag.save()
```

**Context Menu**:
- ✓ Include in Filter (I)
- ✗ Exclude from Filter (E)
- Add Child Tag
- Rename
- Delete

---

## Search & Filtering

### SearchService Integration

**File**: [search_service.py](file:///d:/github/USCore/src/ucorefs/search/service.py)

```python
class SearchQuery:
    text: Optional[str] = None                  # Text search
    filters: Dict[str, Any] = {}                # MongoDB filters
    vector_search: bool = False                 # Enable CLIP/BLIP search
    
    # Filters can include:
    # - tag_ids: List[ObjectId]
    # - album_ids: List[ObjectId]
    # - file_type: str
    # - rating_min: int
```

**Unified Search Flow**:

```
1. User applies filters:
   - Tags: Include ["nature"], Exclude ["urban"]
   - Albums: Include ["Favorites"]
   - Directory: "D:/Photos/2023"
   ↓
2. UI builds SearchQuery:
   query = SearchQuery(
       filters={
           "tag_ids": {"$in": [nature_id]},
           "tag_ids": {"$nin": [urban_id]},
           "album_ids": {"$in": [favorites_id]},
           "path": {"$regex": "^D:/Photos/2023"}
       }
   )
   ↓
3. SearchService.search(query)
   ↓
4. MongoDB query executed:
   db.file_records.find({
       "tag_ids": {"$in": [nature_id], "$nin": [urban_id]},
       "album_ids": {"$in": [favorites_id]},
       "path": {"$regex": "^D:/Photos/2023"}
   })
   ↓
5. Results returned and displayed in FilePaneWidget
```

### Aggregation Queries

**File**: [aggregations.py](file:///d:/github/USCore/src/ucorefs/query/aggregations.py)

```python
class Aggregation:
    @staticmethod
    def group_by_tag() -> List[Dict[str, Any]]:
        """Get file count per tag"""
        return [
            {"$unwind": "$tag_ids"},
            {
                "$group": {
                    "_id": "$tag_ids",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$size_bytes"}
                }
            },
            {"$sort": {"count": -1}}
        ]
    
    @staticmethod
    def group_by_album() -> List[Dict[str, Any]]:
        """Get file count per album"""
        return [
            {"$unwind": "$album_ids"},
            {
                "$group": {
                    "_id": "$album_ids",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}}
        ]
```

**Usage**:
```python
from src.ucorefs.models.file_record import FileRecord

# Get tag statistics
pipeline = Aggregation.group_by_tag()
result = await FileRecord.aggregate(pipeline)
# Result: [{"_id": ObjectId(...), "count": 42, "total_size": 12345678}, ...]
```

---

## Summary

### Key Integration Points

1. **FileRecord is the hub**:
   - `tag_ids` → Links to Tag system
   - `album_ids` → Links to Album system
   - `parent_id` / `root_id` → Links to Directory system

2. **All systems use ObjectId references**:
   - Enables efficient MongoDB queries
   - Supports $in, $nin operators for filtering

3. **Denormalization for performance**:
   - `FileRecord.tags` stores tag full_paths as strings
   - `Tag.file_count` stores cached count
   - `DirectoryRecord.child_count` stores cached count

4. **UI panels emit filter_changed signals**:
   - TagPanel → (include_tag_ids, exclude_tag_ids)
   - AlbumPanel → (include_album_ids, exclude_album_ids)
   - DirectoryPanel → (include_paths, exclude_paths)

5. **SearchService combines all filters**:
   - MongoDB query builder merges all filters
   - Supports hybrid text + vector search
   - Results paginated and scored

### Design Patterns Used

- **Repository Pattern**: `TagManager`, `AlbumManager`, `FSService` abstract database access
- **Service Locator**: All managers registered in locator for DI
- **MVVM**: UI panels bind to ViewModels (TagViewModel, etc.)
- **Observer Pattern**: Signals emitted for filter changes
- **Strategy Pattern**: AlbumManager uses different strategies for manual vs smart albums
- **Lazy Loading**: DirectoryPanel loads children on-demand

### Performance Considerations

1. **Batch Processing**: DirectoryScanner yields batches (50 dirs, 1000 files)
2. **Lazy Loading**: UI only loads visible items
3. **Indexed Fields**: All foreign keys are indexed
4. **Aggregation Pipelines**: Use MongoDB's native aggregation for statistics
5. **Denormalized Data**: Trade storage for read speed

---

## References

- Tag Models: [src/ucorefs/tags/models.py](file:///d:/github/USCore/src/ucorefs/tags/models.py)
- Tag Manager: [src/ucorefs/tags/manager.py](file:///d:/github/USCore/src/ucorefs/tags/manager.py)
- Album Models: [src/ucorefs/albums/models.py](file:///d:/github/USCore/src/ucorefs/albums/models.py)
- Album Manager: [src/ucorefs/albums/manager.py](file:///d:/github/USCore/src/ucorefs/albums/manager.py)
- Directory Models: [src/ucorefs/models/directory.py](file:///d:/github/USCore/src/ucorefs/models/directory.py)
- FS Service: [src/ucorefs/services/fs_service.py](file:///d:/github/USCore/src/ucorefs/services/fs_service.py)
- Search Service: [src/ucorefs/search/service.py](file:///d:/github/USCore/src/ucorefs/search/service.py)
- Tag Panel: [samples/uexplorer/uexplorer_src/ui/docking/tag_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/tag_panel.py)
- Album Panel: [samples/uexplorer/uexplorer_src/ui/docking/album_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/album_panel.py)
- Directory Panel: [samples/uexplorer/uexplorer_src/ui/docking/directory_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/directory_panel.py)
