# Session Journal - Tag/Directory/Album System Research  
**Session Date**: 2025-12-27  
**Session ID**: tag_dir_album_research  
**Objective**: Deep code research on how tag system, directory system, and album system are implemented and work together in USCore

## Progress Log

### Phase: Analysis (Started 17:06)

**Step 1**: Initial repository structure analysis  
- Identified core systems in `src/ucorefs/`
- Found tag system in `src/ucorefs/tags/`
- Found album system in `src/ucorefs/albums/`
- Found directory models in `src/ucorefs/models/directory.py`
- Found UI implementations in `samples/uexplorer/uexplorer_src/ui/docking/`

**Step 2**: Model layer analysis  
- Examined `Tag` model (hierarchical with MPTT structure)
- Examined `Album` model (supports smart albums with queries)
- Examined `DirectoryRecord` model (extends FSRecord)
- Examined `FileRecord` model (has tag_ids, album_ids arrays)

**Step 3**: Manager layer analysis  
- Examined `TagManager` - manages hierarchical tags, synonyms/antonyms
- Examined `AlbumManager` - manages static and smart albums
- Examined `FSService` - filesystem operations, directory hierarchy

**Step 4**: Integration analysis
- Examined `SearchService` - unified search with tag/album filters
- Examined `Aggregation` - statistics by tag, album, etc.
- Examined UI panels - TagPanel, AlbumPanel, DirectoryPanel

**Step 5**: Data flow analysis
- Discovered how file counting works for tags
- Discovered how filtering works (include/exclude)
- Discovered how smart albums use query builders

## Key Findings

1. **Tag System**:
   - Hierarchical MPTT structure for nested tags
   - Supports synonyms and antonyms
   - File count tracking on tags
   - Denormalized tags (both IDs and strings) on files

2. **Directory System**:
   - Hierarchical parent-child relationships
   - Library roots with scan settings
   - Lazy loading for performance
   - Count tracking (child_count, file_count)

3. **Album System**:
   - Manual albums (static file lists)
   - Smart albums (dynamic queries)
   - Hierarchical organization
   - Cover image support

4. **Integration**:
   - All three systems use ObjectId references
   - FileRecord has tag_ids and album_ids arrays
   - SearchService can filter by any combination
   - UI panels support include/exclude filtering

## Next Steps
- Create comprehensive research document
- Document data flow diagrams
- Document count calculation methods
