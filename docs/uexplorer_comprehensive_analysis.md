# UExplorer - Comprehensive Application Analysis

**Document Version**: 1.0  
**Created**: 2025-12-28  
**Research Type**: Deep Code Analysis  

## Table of Contents
1. [Overview](#overview)
2. [Current Implementation Status](#current-implementation-status)
3. [UI Components](#ui-components)
4. [Missing Features & TODOs](#missing-features--todos)
5. [Maintenance Tasks](#maintenance-tasks)

---

## Overview

**UExplorer** is a professional file manager built with PySide6, serving as a comprehensive demonstration of the Foundation template and UCoreFS capabilities. It showcases **73% of Foundation features** (14/19 features).

### Key Features
- Dual-pane file browser with split views
- Dockable panels (Tags, Albums, Relations, Properties)
- Command palette with fuzzy search
- AI-powered detection viewer
- Rules engine UI
- Visual query builder
- Audit logging

---

## Current Implementation Status

### Foundation Integration: 73% Coverage

#### ✅ Core Architecture (80% coverage)
- **ServiceLocator**: DI for 14 systems
- **ApplicationBuilder**: Bootstrap with `with_default_systems()`
- **ConfigManager**: JSON-based configuration
- **DatabaseManager**: MongoDB async ORM (Beanie)
- **BaseSystem**: All services extend BaseSystem
- **TaskSystem**: Background task management
- **CommandBus**: Decoupled file operations
- **JournalService**: Audit logging

#### ✅ GUI Framework (67% coverage)
- **ActionRegistry**: 18 centralized actions
- **Command Palette**: Fuzzy search (`Ctrl+Shift+P`)
- **MenuBuilder**: Declarative menus
- **DockManager**: 4 resizable panels with state persistence
- **BasePanelWidget**: Tags, Albums, Relations, Properties
- **Document Splits**: Dual-pane browser

#### ⚠️ Partial Implementation
- **DialogService**: Custom dialogs exist, but not fully integrated with Foundation DialogService
- **ThemeManager**: Not implemented (uses default Qt styles)

#### ❌ Not Implemented
- **PluginManager**: No extension system
- **UpdateService**: No auto-update functionality
- **TelemetryService**: No analytics

---

## UI Components

### Application Structure

```
samples/uexplorer/uexplorer_src/
├── commands/           # CommandBus pattern
│   └── file_commands.py
├── tasks/              # Background tasks
│   └── scan_task.py
├── models/             # ViewModels
│   └── file_model.py
├── viewmodels/         # Application ViewModels
│   ├── main_viewmodel.py
│   └── document_manager.py
├── ui/
│   ├── actions/        # Centralized actions
│   │   └── action_definitions.py
│   ├── docking/        # Panel widgets
│   │   ├── tag_panel.py
│   │   ├── album_panel.py
│   │   ├── directory_panel.py
│   │   ├── properties_panel.py
│   │   ├── relations_panel.py
│   │   ├── similar_items_panel.py
│   │   ├── annotation_panel.py
│   │   ├── background_panel.py
│   │   └── unified_search_panel.py
│   ├── documents/      # Document views
│   │   ├── file_browser_document.py
│   │   ├── image_viewer_document.py
│   │   ├── split_container.py
│   │   └── split_manager.py
│   ├── dialogs/        # Dialogs
│   │   ├── library_dialog.py
│   │   ├── rule_manager_dialog.py
│   │   └── settings_dialog.py
│   ├── widgets/        # Reusable widgets
│   │   ├── tag_tree.py
│   │   ├── album_tree.py
│   │   ├── file_card_widget.py
│   │   ├── filter_tree_widget.py
│   │   └── metadata_panel.py
│   ├── managers/       # UI managers
│   │   ├── menu_manager.py
│   │   ├── toolbar_manager.py
│   │   ├── selection_manager.py
│   │   └── filter_manager.py
│   └── main_window.py  # Main application window
└── utils/              # Utilities
```

### Main Window
**Location**: [main_window.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/main_window.py)

**Components**:
- Menu bar (File, Edit, View, Tools, Help)
- Toolbar with common actions
- Central document area (split manager)
- 9 dockable panels
- Status bar
- Command palette overlay

**Layout**:
```
┌─────────────────────────────────────────────────┐
│ Menu Bar                                        │
├─────────────────────────────────────────────────┤
│ Toolbar                                         │
├──────┬──────────────────────────────────┬───────┤
│ Dir  │  File Browser (Split View)       │ Props │
│ Tree │  ┌─────────────┬─────────────┐  │ Panel │
│      │  │  Left Pane  │ Right Pane  │  │       │
│ Tags │  │             │             │  │ Relat │
│ Panel│  │             │             │  │ Panel │
│      │  └─────────────┴─────────────┘  │       │
│ Album├──────────────────────────────────┤ Simil │
│ Panel│  Unified Search Panel           │ Panel │
├──────┴──────────────────────────────────┴───────┤
│ Status Bar: 1234 files | Processing...  │
└─────────────────────────────────────────────────┘
```

### Docking Panels

#### 1. **DirectoryPanel** ✅ Complete
**Location**: [directory_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/directory_panel.py)

**Features**:
- Library root navigation
- Lazy loading (loads children on expand)
- Include/exclude filtering
- Refresh button
- File/folder icons

**Signals**:
- `directory_selected(str directory_id, str path)`
- `filter_changed(list include, list exclude)`

#### 2. **TagPanel** ✅ Complete
**Location**: [tag_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/tag_panel.py)

**Features**:
- Hierarchical tag tree (MPTT)
- Drag-drop files to tag
- Include/exclude filtering
- Context menu (Include, Exclude, Add Child, Rename, Delete)
- File count display
- Clear filters button

**Signals**:
- `filter_changed(list include_tag_ids, list exclude_tag_ids)`

#### 3. **AlbumPanel** ✅ Complete
**Location**: [album_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/album_panel.py)

**Features**:
- Album list (manual + smart albums)
- Create new album button
- Include/exclude filtering
- Smart album indicator
- File count display

**Signals**:
- `filter_changed(list include_album_ids, list exclude_album_ids)`
- `smart_album_selected(str album_id, dict query)`

#### 4. **PropertiesPanel** ✅ Complete
**Location**: [properties_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/properties_panel.py)

**Features**:
- File metadata display (path, size, modified date)
- EXIF/XMP metadata
- Thumbnail preview
- Tag list
- Album memberships
- Processing state indicator

#### 5. **RelationsPanel** ✅ Complete
**Location**: [relations_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/relations_panel.py)

**Features**:
- Show file relations (duplicate, similar, sequence)
- Grouped by relation type
- Click to navigate to related file

**TODOs Found**:
```python
# TODO: Implement relation editing
# TODO: Add "Create Relation" button
```

#### 6. **SimilarItemsPanel** ✅ Complete
**Location**: [similar_items_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/similar_items_panel.py)

**Features**:
- Display similar files using perceptual hash
- Threshold slider (0-100)
- Grid view of similar items
- Click to open file

#### 7. **AnnotationPanel** ⚠️ Incomplete
**Location**: [annotation_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/annotation_panel.py)

**Current Features**:
- List annotations for selected file
- Display annotation type and label

**Missing**:
- Annotation editing
- Drawing tools (rectangle, polygon, etc.)
- Class/label management
- Export functionality

#### 8. **BackgroundPanel** ✅ Complete
**Location**: [background_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/background_panel.py)

**Features**:
- Show running background tasks (TaskSystem)
- Task name, progress bar (0-100%), status
- Task history
- Cancel button (for cancelable tasks)

#### 9. **UnifiedSearchPanel** ✅ Complete
**Location**: [unified_search_panel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/docking/unified_search_panel.py)

**Features**:
- Text search input
- Vector search toggle
- Search filters:
  - File type dropdown
  - Rating filter
  - Date range picker
- Execute search button
- Results display

### Document Views

#### FileBrowserDocument ✅ Complete
**Location**: [file_browser_document.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/documents/file_browser_document.py)

**Features**:
- Grid view (QListView with custom delegates)
- List view (QTreeView)
- File cards with thumbnails
- Selection model (multi-select with Ctrl/Shift)
- Drag-drop support
- Context menu

**TODOs Found**:
```python
# TODO: Implement viewport detection for priority queue
# TODO: Add infinite scroll / pagination
```

**Viewport Priority Queue Integration** (missing):
- Currently loads all files in current directory
- Should detect visible files and queue them with HIGH priority
- Need to emit `files_visible(list file_ids)` signal

#### ImageViewerDocument ✅ Complete
**Location**: [image_viewer_document.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/documents/image_viewer_document.py)

**Features**:
- Zoom (fit, 100%, custom)
- Pan (drag with mouse)
- Detection overlay (bounding boxes)
- Annotation overlay (if annotations exist)
- Navigation (prev/next)

#### SplitContainer ✅ Complete
**Location**: [split_container.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/documents/split_container.py)

**Features**:
- Horizontal/vertical splits
- Recursive splits (tree structure)
- Drag splitter to resize
- Close split action

### Dialogs

#### LibraryDialog ✅ Complete
**Location**: [library_dialog.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/dialogs/library_dialog.py)

**Features**:
- Add library root (folder picker)
- Set watch extensions (comma-separated list)
- Set blacklist paths
- Enable/disable scanning

#### RuleManagerDialog ✅ Complete
**Location**: [rule_manager_dialog.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/dialogs/rule_manager_dialog.py)

**Features**:
- List all rules
- Create new rule
- Edit existing rule
- Delete rule
- Test rule against sample files
- Trigger selector (on_import, on_tag, manual)
- Condition builder (visual)
- Action builder (visual)

#### SettingsDialog ✅ Complete
**Location**: [settings_dialog.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/dialogs/settings_dialog.py)

**Features**:
- General settings (theme, language)
- Performance settings (worker count, batch sizes)
- AI settings (GPU selection, model paths)
- Database settings (MongoDB connection)
- Thumbnail settings (cache size, quality)

### Actions

**Location**: [action_definitions.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/actions/action_definitions.py)

**Registered Actions** (18 total):
1. `file.scan_directories` - F5
2. `file.add_library_root`
3. `file.refresh`
4. `edit.copy` - Ctrl+C
5. `edit.paste` - Ctrl+V
6. `edit.delete` - Delete
7. `view.toggle_tag_panel` - Ctrl+1
8. `view.toggle_album_panel` - Ctrl+2
9. `view.toggle_relations_panel` - Ctrl+3
10. `view.toggle_properties_panel` - Ctrl+4
11. `view.split_horizontal` - Ctrl+Shift+H
12. `view.split_vertical` - Ctrl+Shift+V
13. `view.close_split` - Ctrl+Shift+W
14. `tools.command_palette` - Ctrl+Shift+P
15. `tools.settings` - Ctrl+,
16. `tools.keyboard_shortcuts` - Ctrl+?
17. `tools.rule_manager`
18. `help.about`

**TODOs Found**:
```python
# TODO: Add export actions (export to ZIP, export metadata, etc.)
# TODO: Add batch operations (batch tag, batch move, batch rename)
```

### ViewModels

#### MainViewModel ✅ Complete
**Location**: [main_viewmodel.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/viewmodels/main_viewmodel.py)

**Responsibilities**:
- Aggregate filters from all panels (tags, albums, directories)
- Execute unified search query
- Publish `search_results_updated` event
- Handle file selection changes

#### DocumentManager ⚠️ Incomplete
**Location**: [document_manager.py](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/viewmodels/document_manager.py)

**Current Features**:
- Manage open documents (browser, image viewer)
- Create new splits
- Track active document

**TODOs Found**:
```python
# TODO: Implement tab persistence (save/restore open tabs)
# TODO: Add MRU (Most Recently Used) list
```

---

## Missing Features & TODOs

### Critical Missing Features

#### 1. **Viewport Priority Queue Integration** ⚠️ Backend Ready, UI Not Integrated
**Background Ready**: Priority queue implemented in ProcessingPipeline and TaskSystem  
**UI Integration Missing**: FileBrowserDocument doesn't detect visible files  

**Required Changes**:
```python
# In FileBrowserDocument
class FileBrowserDocument(BaseDocument):
    visible_files_changed = Signal(list)  # file_ids
    
    def _on_scroll_or_resize(self):
        """Detect which files are in viewport"""
        viewport = self.file_grid.viewport()
        visible_items = []
        for i in range(self.model.rowCount()):
            rect = self.file_grid.visualRect(self.model.index(i, 0))
            if viewport.rect().intersects(rect):
                file_id = self.model.data(i, Qt.UserRole)
                visible_items.append(file_id)
        
        self.visible_files_changed.emit(visible_items)

# In MainWindow
def _on_visible_files_changed(self, file_ids: list):
    """Queue visible files with high priority"""
    pipeline = self.locator.get_system(ProcessingPipeline)
    
    for file_id in file_ids:
        file = await FileRecord.get(file_id)
        if file.processing_state == ProcessingState.PHASE2_PENDING:
            await pipeline.enqueue_phase2([file_id], priority=0)  # HIGH
        elif file.processing_state == ProcessingState.PHASE3_PENDING:
            await pipeline.enqueue_phase3([file_id], priority=0)  # HIGH
```

**Impact**: User sees thumbnails and metadata for visible files first (better UX)

#### 2. **Annotation Editing UI** ❌ Missing
**Current State**: AnnotationPanel displays existing annotations  
**Missing**: Drawing tools, editing, class management

**Required**:
- Drawing overlay in ImageViewerDocument (rectangle, polygon, point tools)
- Mouse handlers for creating/editing annotations
- Annotation class picker
- Save/discard annotation changes

#### 3. **Relation Editing** ❌ Missing
**Current State**: RelationsPanel displays relations (read-only)  
**Missing**: Create/delete relations manually

**Required**:
- "Create Relation" button in RelationsPanel
- Relation type picker (duplicate, similar, sequence, etc.)
- Target file picker
- Delete relation action

#### 4. **Export Actions** ❌ Missing
**Current State**: No export functionality  

**Proposed Actions**:
- Export selected files to ZIP
- Export metadata to CSV/JSON
- Export search results
- Export annotations (COCO, YOLO, Pascal VOC formats)
- Export detection results

#### 5. **Batch Operations** ❌ Missing
**Current State**: Operations work on single files only  

**Proposed**:
- Batch tag application
- Batch move to folder
- Batch rename (with patterns)
- Batch rating change
- Batch delete

#### 6. **Tab Persistence** ❌ Missing
**Current State**: Open documents not saved on app close  
**TODOs Found**: In DocumentManager

**Required**:
- Save open document list to config
- Restore on app start
- MRU (Most Recently Used) list

#### 7. **Infinite Scroll / Pagination** ❌ Missing
**Current State**: Loads all files in directory (slow for large dirs)  
**TODOs Found**: In FileBrowserDocument

**Required**:
- Lazy loading (load files as user scrolls)
- Pagination controls
- Virtual scrolling for very large collections

### Non-Critical Gaps

#### 8. **Theme Manager** ❌ Not Implemented
**Current State**: Uses default Qt styles  
**Foundation Feature**: ThemeManager available but not integrated

**Impact**: Limited customization options

#### 9. **Plugin System** ❌ Not Implemented
**Current State**: No extension mechanism  
**Foundation Feature**: PluginManager available but not used

**Potential Use Cases**:
- Custom extractors
- Custom file drivers
- Custom actions
- Custom panels

#### 10. **Auto-Update** ❌ Not Implemented
**Foundation Feature**: UpdateService available but not integrated

#### 11. **Analytics** ❌ Not Implemented
**Foundation Feature**: TelemetryService available but not integrated

---

## Maintenance Tasks

### High Priority

#### 1. **Implement Viewport Priority Queue UI Integration**
**Status**: Backend ready, UI integration deferred  
**Effort**: Medium (2-3 hours)  
**Impact**: High (significantly improves UX)

**Steps**:
1. Add scroll/resize handlers to FileBrowserDocument
2. Calculate visible file IDs in viewport
3. Emit `visible_files_changed` signal
4. Connect to MainWindow to queue files with HIGH priority

#### 2. **Add Export Actions**
**Status**: Missing  
**Effort**: Medium (4-6 hours)  
**Impact**: High (users need export functionality)

**Actions to Implement**:
- Export to ZIP
- Export metadata to CSV/JSON
- Export search results
- Export annotations

#### 3. **Implement Batch Operations**
**Status**: Missing  
**Effort**: Medium (4-8 hours)  
**Impact**: High (productivity feature)

**Operations to Implement**:
- Batch tag
- Batch move
- Batch rename
- Batch rating

### Medium Priority

#### 4. **Complete Annotation Editing**
**Status**: Viewer exists, editing missing  
**Effort**: High (1-2 days)  
**Impact**: Medium (depends on annotation use case)

**Components Needed**:
- Drawing tools overlay
- Mouse event handlers
- Annotation class picker
- Save/discard UI

#### 5. **Implement Relation Editing**
**Status**: Viewer exists, editing missing  
**Effort**: Low (2-4 hours)  
**Impact**: Medium

**Features Needed**:
- "Create Relation" dialog
- Relation type picker
- Target file picker
- Delete relation button

#### 6. **Add Tab Persistence**
**Status**: TODO in DocumentManager  
**Effort**: Low (2-3 hours)  
**Impact**: Medium (convenience feature)

**Implementation**:
- Save document state to JSON
- Restore on app start
- Add MRU list dialog

### Low Priority

#### 7. **Implement Infinite Scroll**
**Status**: TODO in FileBrowserDocument  
**Effort**: Medium (4-6 hours)  
**Impact**: Low (only matters for very large directories)

#### 8. **Integrate ThemeManager**
**Status**: Foundation feature not used  
**Effort**: Low (1-2 hours)  
**Impact**: Low (visual polish)

#### 9. **Add Plugin System**
**Status**: Foundation feature not used  
**Effort**: High (1-2 days)  
**Impact**: Low (advanced feature)

---

## Architecture Strengths

### What's Working Well

1. **MVVM Pattern**: Clean separation between UI and business logic
2. **Signals/Slots**: Decoupled communication between components
3. **Action Registry**: Centralized action management (easy keyboard shortcut customization)
4. **Command Palette**: Discoverable features (user can search all actions)
5. **Panel Docking**: Flexible workspace layout
6. **Split Views**: Power-user feature (dual-pane browser)
7. **Lazy Loading**: Efficient for large directory trees
8. **Drag-Drop**: Intuitive file tagging

### Design Patterns Used

- **MVVM**: ViewModels separate UI from business logic
- **Observer**: Signals/slots for event handling
- **Command**: ActionRegistry for undo/redo-capable actions
- **Repository**: FSService, TagManager, AlbumManager
- **Service Locator**: Dependency injection
- **Facade**: MainViewModel aggregates filters from multiple panels

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+P` | Command Palette |
| `Ctrl+1` | Toggle Tag Panel |
| `Ctrl+2` | Toggle Album Panel |
| `Ctrl+3` | Toggle Relations Panel |
| `Ctrl+4` | Toggle Properties Panel |
| `Ctrl+Shift+H` | Split Horizontal |
| `Ctrl+Shift+V` | Split Vertical |
| `Ctrl+Shift+W` | Close Split |
| `Ctrl+,` | Settings |
| `Ctrl+?` | Keyboard Shortcuts |
| `F5` | Scan Directories |
| `Ctrl+C` | Copy |
| `Ctrl+V` | Paste |
| `Delete` | Delete File |

---

## Testing

**Location**: `samples/uexplorer/tests/`

**Test Coverage**:
- Unit tests for ViewModels
- Integration tests for background tasks
- UI tests (limited)

**Gaps**:
- No automated UI tests (consider using `pytest-qt`)
- Limited integration testing
- No performance benchmarks

---

## Dependencies

### Core
- `PySide6` - Qt6 Python bindings
- `qasync` - Asyncio event loop for Qt

### Backend (from UCore FS)
- `motor` - Async MongoDB
- `beanie` - Async ORM
- `pydantic` - Data validation
- `loguru` - Logging

### AI (optional)
- `torch` - PyTorch
- `transformers` - Hugging Face models
- `imagehash` - Perceptual hashing
- `faiss-cpu` or `faiss-gpu` - Vector search

---

## References

### Documentation
- [UExplorer README](file:///d:/github/USCore/samples/uexplorer/README.md)
- [UCore FS README](file:///d:/github/USCore/src/ucorefs/README.md)
- [Tag/Directory/Album Systems](file:///d:/github/USCore/docs/tag_directory_album_systems.md)

### Source Code
- **Main Window**: `samples/uexplorer/uexplorer_src/ui/main_window.py`
- **Panels**: `samples/uexplorer/uexplorer_src/ui/docking/`
- **Documents**: `samples/uexplorer/uexplorer_src/ui/documents/`
- **Actions**: `samples/uexplorer/uexplorer_src/ui/actions/action_definitions.py`
