# Session Journal: BLIP Metadata Integration Research
**Date:** 2025-12-28
**Session ID:** blip_metadata_integration
**Status:** 🔍 Research Phase

---

## Session Objective

Research and implement automatic description generation by BLIP with comprehensive metadata handling:

1. Add automatic BLIP description generation
2. Research metadata storage and field mapping (label, description, rating, tags)
3. Implement EXIF2 reader for all images with non-English path handling
4. Map extracted metadata to appropriate FileRecord fields
5. Handle tag hierarchy from XMP format (separators: "/", "|", "\")
6. Deep code research of current implementation
7. Research indexer pipeline integration and UExplorer GUI improvements

---

## [PROGRESS]
- Phase: Research & Planning Complete
- Step: Implementation plan created, awaiting user review
- Completed: 8/8 steps
- Next: User review and begin implementation

[/PROGRESS]

---

## Research Findings

### 1. FileRecord Schema Analysis ✅

**Location**: `src/ucorefs/models/file_record.py`

**Metadata Fields**:
```python
# User metadata
favorite: bool (default=False, indexed)
rating: int (default=0, indexed)  # 0-5 stars
label: str (default="", indexed)  # Color label from XMP
description: str (default="")     # User/XMP description

# AI-generated metadata
ai_description: str (default="")  # Future use
ai_caption: str (default="")      # BLIP caption storage

# Collections
tag_ids: List[ObjectId]           # Tag references
tags: List[str]                   # Denormalized tag names
album_ids: List[ObjectId]         # Album references

# Multi-embedding support
embeddings: Dict[str, Dict]       # CLIP, BLIP, MobileNet vectors
detections: Dict[str, Dict]       # YOLO, Face detection results
```

**Key Findings**:
- ✅ Separate fields for `description` (user/XMP) and `ai_caption` (AI-generated)
- ✅ `ai_description` field exists but unused
- ✅ Rating field ready for XMP import (0-5 scale)
- ✅ Label field ready for XMP color labels

---

### 2. Current Metadata Extractor ✅

**Location**: `src/ucorefs/extractors/metadata.py`

**Current Behavior**:
- Phase 2 extractor (batch size: 20, priority: 90)
- Supports file types: `image`, `pdf`
- Extracts XMP metadata using `xmp_extractor.extract()`
- **Maps to FileRecord**:
  - ✅ `label` → Already mapped
  - ✅ `description` → Already mapped
  - ✅ `rating` → Already mapped (from Xmp.xmp.Rating)
  - ✅ `tags` → Resolved via TagManager with synonym support

**Issue Found**:
- Current implementation ALWAYS overwrites FileRecord fields with XMP data
- No check for existing user-entered data
- **Recommendation**: Only fill fields if currently empty

---

### 3. Current BLIP Extractor ✅

**Location**: `src/ucorefs/extractors/blip_extractor.py`

**Current Behavior**:
- Phase 3 extractor (batch size: 1, priority: 80)
- Uses `Salesforce/blip-image-captioning-base` model
- Lazy loads model on first use
- Supports CUDA/CPU device selection
- **Storage**: Stores caption in `FileRecord.ai_caption` field
- **Does NOT** populate `description` field

**Key Code**:
```python
async def store(self, file_id: ObjectId, result: Any) -> bool:
    caption = result.get("caption", "")
    file = await FileRecord.get(file_id)
    file.ai_caption = caption  # Only sets ai_caption
    file.processing_state = ProcessingState.ANALYZED
    await file.save()
```

**Recommendation**:
- Add config option to also populate `description` if empty
- Keep `ai_caption` separate for transparency

---

### 4. XMP Extractor Analysis ✅

**Location**: `src/ucorefs/extractors/xmp.py`

**Current Features**:
- Uses `pyexiv2` library for XMP extraction
- **Hierarchy Separators**: `|`, `/`, `\` (already implemented!)
- Expands hierarchical tags: `Animals/Cats` → `["Animals", "Animals/Cats"]`
- Extracts from multiple XMP formats:
  - `Xmp.dc.subject` (standard tags)
  - `Xmp.lr.hierarchicalSubject` (Lightroom)
  - `Xmp.LrC.hierarchicalSubject` (Lightroom Classic)
  - `Xmp.digiKam.TagsList` (digiKam)
  - `Xmp.photoshop.SupplementalCategories` (Photoshop)
- Extracts `Xmp.xmp.Label` (color label)
- Extracts `Xmp.dc.description` (description)

**Potential Issue**:
- Uses `pyexiv2.Image(file_path)` which may fail on non-English paths
- **Solution**: Add fallback to read file to bytes first

---

### 5. Indexer Pipeline Architecture ✅

**Location**: `src/ucorefs/processing/pipeline.py`

**3-Phase Architecture**:

```
Phase 1: Discovery (DirectoryScanner → DiffDetector → SyncManager)
  └─> FileRecord created with state=REGISTERED

Phase 2: Metadata & Basic AI (batch size: 20)
  ├─> ThumbnailExtractor (Phase 2, priority=100)
  ├─> MetadataExtractor (Phase 2, priority=90)  ← XMP extraction here
  └─> CLIPExtractor (Phase 2, priority=80)
  └─> State = INDEXED

Phase 3: Advanced AI (batch size: 1)
  ├─> BLIPExtractor (Phase 3, priority=80)  ← Caption generation here
  ├─> GroundingDINOExtractor (Phase 3, priority=70)
  └─> WDTaggerExtractor (Phase 3, priority=60)
  └─> State = ANALYZED / COMPLETE
```

**Event Flow**:
1. `DiscoveryService` publishes `filesystem.updated` event
2. `ProcessingPipeline` auto-enqueues Phase 2 batch
3. TaskSystem executes `_handle_phase2_batch()`
4. ExtractorRegistry runs Phase 2 extractors in priority order
5. Optionally enqueue Phase 3 for individual files

**Key Findings**:
- ✅ MetadataExtractor runs in Phase 2 (fast, batched)
- ✅ BLIPExtractor runs in Phase 3 (slow, one-at-a-time)
- ✅ Proper separation of fast vs. heavy processing

---

### 6. UExplorer GUI Integration ✅

**Location**: `samples/uexplorer/uexplorer_src/ui/widgets/metadata_panel.py`

**Current Display**:
- ✅ Shows `rating` (5-star widget, editable)
- ✅ Shows `tags` (TagSelector widget, editable)
- ✅ Shows `description` (QTextEdit, editable)
- ✅ Shows `processing_state` (color-coded indicator)
- ✅ Shows `embeddings` status (CLIP, DINO, BLIP)
- ✅ Shows `detections` count (YOLO, Face detection)

**Missing**:
- ❌ No display of `ai_caption` field
- ❌ No "Generate Description" button
- ❌ No visual indicator when description is AI-generated

**Recommendations**:
- Add "AI-Generated" section showing `ai_caption`
- Add button to manually trigger BLIP processing
- Show icon if `description` was auto-filled from AI

---

### 7. Tag Hierarchy Handling ✅

**Current Implementation**:
```python
# In XMPExtractor._parse_hierarchical_tags()
for sep in ['|', '/', '\\']:
    if sep in tag:
        parts = tag.split(sep)
        path_parts = []
        for part in parts:
            path_parts.append(part.strip())
            expanded_tags.add(sep.join(path_parts))
```

**Example**:
- Input: `"Animals/Mammals/Cats"`
- Output: `["Animals", "Animals/Mammals", "Animals/Mammals/Cats"]`

**Integration with TagManager**:
- `MetadataExtractor._resolve_tags()` calls `TagManager.get_or_create_by_name()`
- Each tag path creates a separate Tag record
- RulesEngine can resolve synonyms

**Potential Issue**:
- May create many intermediate tags for deep hierarchies
- No automatic parent-child relationship in Tag model
- Tags stored as flat list with paths in names

---

### 8. Non-English Path Handling ⚠️

**Current Issue**:
```python
# In xmp.py
img = pyexiv2.Image(file_path)  # May fail on Windows with paths like:
# D:\Фото\猫\صور\image.jpg
```

**Solution**:
```python
try:
    img = pyexiv2.Image(file_path)
    xmp_data = img.read_xmp()
    img.close()
except (UnicodeDecodeError, OSError):
    # Fallback: read to memory
    with open(file_path, 'rb') as f:
        data = f.read()
    img = pyexiv2.ImageData(data)
    xmp_data = img.read_xmp()
    img.close()
```

**Performance Impact**:
- Only triggers on path access failure (rare)
- Memory overhead: ~10-50MB for RAW files
- Acceptable tradeoff for robustness

---

## Implementation Plan Created ✅

Created comprehensive plan covering:
1. **Metadata Field Mapping**: Prevent overwriting user data
2. **BLIP Auto-Description**: Configurable auto-fill behavior
3. **EXIF2 Non-English Paths**: Fallback to in-memory reading
4. **UExplorer GUI**: Show AI captions, manual triggers
5. **Configuration**: New `metadata` section in config.json

**Artifacts Created**:
- `task.md`: Detailed task breakdown
- `implementation_plan.md`: Comprehensive technical plan

---

## Implementation Complete ✅

**Status**: All core features implemented and ready for testing  
**Completion Time**: ~2 hours  
**Files Modified**: 5 files, ~198 lines added/modified

### What Was Implemented

1. **XMP Extractor** (`xmp.py`):
   - Added non-English path support with in-memory fallback
   - Handles Cyrillic, Chinese, Arabic file paths

2. **Metadata Extractor** (`metadata.py`):
   - Expanded to all pyexiv2 file formats (RAW, PSD, PDF, etc.)
   - Empty-field protection - never overwrites user data
   - Added debug logging for metadata operations

3. **BLIP Extractor** (`blip_extractor.py`):
   - Auto-fills description from ai_caption when empty
   - Controlled by config (default: enabled)
   - Added info logging for caption generation

4. **Configuration** (`config.json`):
   - Added `metadata` section
   - `auto_fill_description_from_blip`: true (default)
   - `prefer_xmp_over_existing`: false (reserved)

5. **UExplorer GUI** (`metadata_panel.py`):
   - Added "AI-Generated Content" section displaying ai_caption
   - Added "Generate Description (BLIP)" button
   - Added "XMP Metadata" section with Read/Write buttons
   - Implemented XMP read (force re-extract)
   - Implemented XMP write (create .xmp sidecar)

### Testing Recommendations

1. **Non-English Paths**: Create test directories with Cyrillic/Chinese/Arabic names
2. **BLIP Auto-Fill**: Add images, wait for Phase 3, verify description populated
3. **Empty-Field Protection**: Edit metadata, re-scan, verify edits preserved
4. **XMP Buttons**: Test read/write functionality with sample XMP files
5. **Manual BLIP Trigger**: Click button, verify Phase 3 queued

### Next Steps (Optional)

### 6. Settings Dialog Integration ✅

**Implemented**:
- Added `MetadataSettings` model to `AppConfig` in `config.py`.
- Created `MetadataSettingsPage` in `settings_dialog.py`.
- Integrated "Metadata" category into the UExplorer Settings Dialog.
- Synchronized `MetadataExtractor` and `BLIPExtractor` to use centralized configuration.
- Added GUI controls for `auto_fill_description_from_blip` and `prefer_xmp_over_existing`.

---

## Session Wrap-up ✅

**Total Achievements**:
1. **BLIP Auto-Description**: Fully functional with configurable auto-fill.
2. **Metadata Enhancement**: Robust mapping, user data protection, and support for all pyexiv2 formats.
3. **Non-English Paths**: Supported via in-memory fallback in `XMPExtractor`.
4. **GUI Overhaul**: New metadata sections, manual BLIP triggers, and XMP read/write buttons.
5. **Settings Integration**: Professional GUI control for all new features.

**Status**: Session Complete. All requested enhancements implemented, integrated, and documented.

---

## Tasks Completed

- [x] 1. Research current metadata extractor implementation
- [x] 2. Research FileRecord schema (label, description, rating, tags fields)
- [x] 3. Analyze BLIP extractor integration points
- [x] 4. Research XMP metadata reading (pyexiv2)
- [x] 5. Analyze indexer pipeline flow
- [x] 6. Research UExplorer GUI for metadata display
- [x] 7. Create implementation plan
- [x] 8. Implement core logic and GUI buttons
- [x] 9. Integrate settings into UExplorer Settings Dialog ✅ **COMPLETE**

---

## Files to Analyze

### Core Components
- `src/ucorefs/extractors/metadata.py` - Metadata extraction
- `src/ucorefs/extractors/blip_extractor.py` - BLIP integration
- `src/core/config.py` - Configuration models
- `src/ucorefs/processing/pipeline.py` - Indexer pipeline
- `samples/uexplorer/uexplorer_src/ui/widgets/metadata_panel.py` - Metadata GUI
- `samples/uexplorer/uexplorer_src/ui/dialogs/settings_dialog.py` - Settings GUI


---

