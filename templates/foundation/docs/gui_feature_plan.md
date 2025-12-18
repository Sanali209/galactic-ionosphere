# Advanced GUI Framework - Revised Plan

**Status**: 📋 Planning Phase (Revised)
**Integration**: Foundation Template v1.0 (runs from `main.py`)
**Priority**: Document Management & Settings  
**Testing**: 85% pass rate required per phase
**Inspiration**: VS Code editor layout

---

## 🎯 Overview

This plan focuses on creating a robust document management system with flexible layouts, panel state persistence, and a standard settings dialog. **File explorer is deferred** for future implementation with custom file handling.

## 🏗️ Architecture Overview

### Current Foundation (Already Implemented)
- ✅ `QMainWindow` base with basic docking
- ✅ MVVM pattern with `BaseViewModel`
- ✅ Service Locator integration
- ✅ Command Bus for UI actions
- ✅ Reactive configuration system

### Immediate Priorities (This Phase)

```
┌──────────────────────────────────────────────────────┐
│  File  Edit  View  Tools  Window  Help        [_][X] │
├──────────────────────────────────────────────────────┤
│  [New] [Open] [Save] | [Split H] [Split V]  Search  │
├────────────────┬────────────────┬────────────────────┤
│  Document 1    │  Document 2    │  Properties Panel  │
│  ═════════════ │  ═════════════ │  ┌───────────────┐ │
│                │                │  │ Object Info   │ │
│  Content...    │  Content...    │  │               │ │
├────────────────┴────────────────┤  │               │ │
│       Document 3                │  └───────────────┘ │
│          │                            │                  │
├──────────┴────────────────────────────┴──────────────────┤
│              Output Panel / Terminal                      │
└──────────────────────────────────────────────────────────┘
```

---

---

## 📦 Core Features (Priority Order)

### 1. Flexible Document Management System

#### 1.1 Unlimited Document Splits
**Primary Goal**: Allow users to view and edit multiple documents simultaneously in any configuration.

**Features**:
- **Side-by-side splits** - Place documents horizontally next to each other
- **Top-down splits** - Stack documents vertically
- **Nested splits** - Unlimited split depth (e.g., 2 docs side-by-side, each split top-down)
- **Drag to split** - Drag tab to edge to create split in that direction
- **Resize splits** - Draggable splitter handles with snap-to-equal
- **Close split** - When last tab in split closes, merge with neighboring split
- **Tab management per split** - Each split area has its own tab bar

**User Scenarios**:
```
Scenario 1: Compare two files side-by-side
┌─────────────┬─────────────┐
│  File A     │  File B     │
│  (original) │  (modified) │
└─────────────┴─────────────┘

Scenario 2: Reference docs while coding
┌──────────────┬──────────┐
│  Code.py     │ Docs.md  │
│              │          │
├──────────────┤          │
│  Terminal    │          │
└──────────────┴──────────┘

Scenario 3: Complex multi-file workflow
┌──────┬──────┬──────┐
│ SQL  │ UI   │ Data │
├──────┴──────┤      │
│   Results   │      │
└─────────────┴──────┘
```

**Integration with Template**:
- Central widget becomes `SplitDocumentManager`
- Each split is a `SplitContainer` with tab bar
- Document state managed by `DocumentViewModel`
- Split layout saved in `ConfigManager.data.ui.layout_state`
- Commands: `SplitHorizontalCommand`, `SplitVerticalCommand`, `CloseSplitCommand`

**Implementation Classes**:
```python
src/ui/documents/
├── split_manager.py         # Manages split tree structure
├── split_container.py       # Container for tabs in one split
├── document_tab_bar.py      # Tab bar with drag/drop
├── document_view.py         # Base class for document widgets
├── splitter_handle.py       # Custom QSplitter with snap guides
└── layout_state.py          # Serialize/restore split tree
```

---

### 2. Panel State Management

#### 2.1 Persistent Panel Positions
**Goal**: Remember which panels are open, where they're docked, and their sizes.

**Features**:
- **Auto-save state** - On application close, save:
  - Which panels are visible
  - Dock position (left/right/bottom/floating)
  - Panel size (width/height)
  - Tab order if multiple panels in same dock area
- **Auto-restore** - On startup, recreate exact layout
- **Layout presets** - Save/load named layouts ("Coding", "Review", "Debug")
- **Reset to defaults** - Menu: View → Reset Layout

**Storage Format** (in `config.json`):
```json
{
  "ui": {
    "panels": {
      "properties": {"visible": true, "area": "right", "width": 300},
      "output": {"visible": true, "area": "bottom", "height": 200},
      "outline": {"visible": false}
    },
    "panel_order": ["properties", "inspector"],
    "layout_preset": "default"
  }
}
```

**Integration with Template**:
- `DockManager` class manages all panel lifecycle
- On shutdown: `DockManager.save_state()` → `ConfigManager`
- On startup: `DockManager.restore_state()` ← `ConfigManager`
- Panel registry: `DockManager.register_panel(name, panel_class)`

**Implementation Classes**:
```python
src/ui/docking/
├── dock_manager.py          # Central panel lifecycle manager
├── dock_state.py            # State serialization
├── panel_base.py            # Base class for all panels
└── layout_presets.py        # Save/load named layouts
```

---

### 3. Standard Menu Bar

#### 3.1 Complete Menu System
**Goal**: Provide all application actions through discoverable menus.

**Menu Structure**:
```
File
├── New Document          Ctrl+N
├── Open...               Ctrl+O
├── Open Recent          ▶
├── ───────────────
├── Save                  Ctrl+S
├── Save As...            Ctrl+Shift+S
├── Save All
├── ───────────────
├── Close                 Ctrl+W
├── Close All
├── ───────────────
├── Exit                  Alt+F4

Edit
├── Undo                  Ctrl+Z
├── Redo                  Ctrl+Y
├── ───────────────
├── Cut                   Ctrl+X
├── Copy                  Ctrl+C
├── Paste                 Ctrl+V
├── ───────────────
├── Find...               Ctrl+F
├── Replace...            Ctrl+H

View
├── Panels               ▶
│   ├── □ Properties
│   ├── □ Output
│   └── □ Outline
├── ───────────────
├── Split Horizontal      Ctrl+\
├── Split Vertical        Ctrl+Shift+\
├── ───────────────
├── Reset Layout
├── Layout Presets       ▶

Tools
├── Settings...           Ctrl+,
├── Command Palette...    Ctrl+Shift+P
├── ───────────────
├── Task Manager
└── Journal Viewer

Window
├── Next Document         Ctrl+Tab
├── Previous Document     Ctrl+Shift+Tab
├── ───────────────
└── Documents            ▶ (list of open docs)

Help
├── Documentation
├── Keyboard Shortcuts
├── ───────────────
└── About
```

**Integration with Template**:
- All menu actions dispatch commands via `CommandBus`
- Menu state updates based on context (e.g., Save enabled only if document modified)
- Recent files list stored in `ConfigManager.data.ui.recent_files`
- Use `QAction` pool managed by `ActionRegistry`

**Implementation Classes**:
```python
src/ui/menus/
├── menu_builder.py          # Build menus from command registry
├── action_registry.py       # QAction pool
├── recent_files.py          # Manage recent file list
└── context_manager.py       # Enable/disable actions based on state
```

---

### 4. Settings Dialog

#### 1.1 Dynamic Dock Management
**Current**: Static 3 docks (Explorer, Properties, Output)
**Proposed**:
- **Drag-and-drop repositioning** of dock widgets
- **Tabbed docking** - multiple panels in same area
- **Floating windows** - detach docks to separate windows
- **Nested docking** - create custom layouts with splitters
- **Layout presets** - save/restore custom workspace configurations

**Integration with Template**:
- Extend `MainWindow.create_docks()` to use `DockManager` class
- Store layout state in `ConfigManager` (`layout.workspace_state`)
- Use `CommandBus` for dock operations (Show/Hide/Reset Layout)

**Implementation Classes**:
```
src/ui/docking/
├── dock_manager.py       # Central dock orchestrator
├── dock_area.py          # Custom QDockWidget with enhanced features
├── dock_titlebar.py      # Custom title bar with actions
└── layout_serializer.py  # Save/restore layouts to JSON
```

---

### 2. Multi-Document Interface (MDI)

#### 2.1 Tabbed Document Area
**Inspiration**: VS Code editor tabs
**Features**:
- **Tab management**: Close, close others, close all, pin tabs
- **Tab grouping**: Split editor vertically/horizontally
- **Tab preview**: Hover to see document preview
- **Dirty indicator**: Modified files marked with dot
- **Tab context menu**: Close, reveal in explorer, copy path

**Integration with Template**:
- Central widget becomes `DocumentManager`
- Each document is a `DocumentView` (extends `QWidget`)
- Document lifecycle managed by `DocumentViewModel`
- Auto-save using `TaskSystem` background jobs

**Implementation Classes**:
```
src/ui/documents/
├── document_manager.py       # Tab container & split view logic
├── document_view.py          # Base class for all document types
├── document_viewmodel.py     # MVVM for document state
├── tab_bar.py                # Custom tab bar with enhanced features
└── document_factory.py       # Registry for document types
```

**Document Types** (Extensible via Plugin System):
- `TextDocumentView` - Code/text editor
- `ImageDocumentView` - Image viewer
- `TableDocumentView` - Data grid
- `MarkdownDocumentView` - Live preview

---

### 3. Standardized Settings Dialog

#### 3.1 VS Code-Style Preferences
**Features**:
- **Category tree** on left (General, Editor, Extensions, etc.)
- **Search/filter** settings by keyword
- **Live preview** - changes apply immediately
- **Reset to defaults** per category
- **User vs. System** settings differentiation
- **JSON editor** fallback for power users

**Integration with Template**:
- Settings backed by `ConfigManager` Pydantic models
- Changes emit `on_changed` events for reactive updates
- Validation using Pydantic field constraints
- Settings plugins via `DriverManager`

**Implementation Classes**:
```
src/ui/settings/
├── settings_dialog.py       # Main modal dialog
├── settings_model.py        # Tree model for categories
├── settings_widgets.py      # Custom editors (color picker, hotkey, etc.)
├── settings_search.py       # Fuzzy search across settings
└── settings_exporter.py     # Import/export settings JSON
```

**Settings Categories**:
```
General
├── Appearance (Theme, Font Size)
├── Language & Region
└── Startup Behavior

Editor
├── Font & Colors
├── Code Formatting
└── Keybindings

Database
├── Connection Settings
├── Query Timeout
└── Cache Size

Extensions
└── Installed Plugin Settings
```

---

### 5. Basic Panel Framework

#### 5.1 Minimal Panel System
**Goal**: Allow custom panels to be added easily.

**Features**:
- **Base panel class** - All panels inherit from `BasePanelWidget`
- **Registration** - `panel_registry.register("output", OutputPanel)`
- **Lifecycle hooks** - `on_show()`, `on_hide()`, `on_update()`
- **Context awareness** - Panels can react to active document change

**Starter Panels** (minimal implementations):
- **Output Panel** - Display application logs
- **Properties Panel** - Show selected object metadata (placeholder)

**Deferred Panels** (future implementation):
- ~~File Explorer~~ - Planned for later with custom file handling
- ~~Search~~ - Find in files (future)
- ~~Outline~~ - Document structure (future)

---

## 🔗 Integration with Core Systems

### Document Management ↔ MVVM
- Each document has a `DocumentViewModel`
- ViewModel exposes: `is_modified`, `file_path`, `title`
- Signals: `content_changed`, `saved`, `closed`

### Split Layout ↔ ConfigManager
- Split tree serialized to JSON
- Pydantic model: `SplitLayoutState`
- Auto-save on every split/close action

### Menus ↔ CommandBus
- Every menu action = `ICommand`
- `SaveDocumentCommand`, `SplitHorizontalCommand`, etc.
- Commands logged to `JournalService`

### Settings ↔ ConfigManager
- Direct Pydantic model editing
- Changes emit `on_changed` events
- UI widgets auto-update

### Panels ↔ ServiceLocator
- Panels registered as lightweight components
- Access services: `self.locator.get_system(TaskSystem)`

---

## 📋 Revised Implementation Phases

### Phase 1: Document Split System (Week 1-2)
**Deliverables**:
- [ ] `SplitManager` with tree structure
- [ ] `SplitContainer` with tab bar
- [ ] Drag tab to split functionality
- [ ] Serialize/restore split layout
- [ ] Basic document types (Text, Placeholder)

**Testing Requirements** (Target: 85% pass rate):

#### Unit Tests (`tests/test_split_manager.py`)
- [ ] Test split tree creation (horizontal/vertical)
- [ ] Test split removal and merge logic
- [ ] Test split navigation (find split by ID)
- [ ] Test split resizing
- [ ] Test edge cases (single split, empty splits)

#### Unit Tests (`tests/test_document_view.py`)
- [ ] Test document creation with ViewModel
- [ ] Test document state (modified flag)
- [ ] Test document save/load simulation
- [ ] Test document signals (content_changed)

#### Unit Tests (`tests/test_layout_state.py`)
- [ ] Test split tree serialization to JSON
- [ ] Test deserialization and recreation
- [ ] Test round-trip (serialize → deserialize → compare)
- [ ] Test complex nested layouts (3+ levels)

#### Integration Tests (`tests/test_split_integration.py`)
- [ ] Test create 5 documents, split into 3 areas
- [ ] Test drag tab between splits
- [ ] Test close all tabs in split → auto-merge
- [ ] Test save layout → restart app → verify restore
- [ ] Test split + panel state persistence together

**Manual Testing Checklist**:
- [ ] Open app, create 2 documents side-by-side
- [ ] Split one document vertically (3 total)
- [ ] Close app, reopen → verify 3-split layout restored
- [ ] Drag splitter handles → verify resize
- [ ] Drag tab to new position → verify reorder

**Success Criteria**: Minimum 17/20 tests passing (85%)

---

### Phase 2: Panel State & Menu (Week 3)
**Deliverables**:
- [ ] `DockManager` with state persistence
- [ ] Complete menu bar (File/Edit/View/Tools/Window/Help)
- [ ] `ActionRegistry` for menu actions
- [ ] Recent files list
- [ ] Output Panel (basic log viewer)

**Testing Requirements** (Target: 85% pass rate):

#### Unit Tests (`tests/test_dock_manager.py`)
- [ ] Test panel registration
- [ ] Test panel show/hide
- [ ] Test dock state serialization
- [ ] Test panel position (left/right/bottom)
- [ ] Test panel size persistence

#### Unit Tests (`tests/test_menu_builder.py`)
- [ ] Test menu creation from command registry
- [ ] Test action enable/disable based on context
- [ ] Test keyboard shortcuts assignment
- [ ] Test recent files list (add/remove)
- [ ] Test menu hierarchy (File → Recent →)

#### Unit Tests (`tests/test_action_registry.py`)
- [ ] Test action registration
- [ ] Test action retrieval by name
- [ ] Test action trigger → command dispatch
- [ ] Test action context (enabled only when doc open)

#### Integration Tests (`tests/test_menu_integration.py`)
- [ ] Test File → Open → dispatches OpenCommand
- [ ] Test View → Properties → toggles panel visibility
- [ ] Test recent files updates after open
- [ ] Test panel state saves on close

#### Integration Tests (`tests/test_panel_persistence.py`)
- [ ] Test open Properties panel (right, 300px wide)
- [ ] Test close app → reopen → verify panel restored
- [ ] Test hide panel → close → reopen → verify hidden
- [ ] Test multiple panels → save/restore order

**Manual Testing Checklist**:
- [ ] Open Properties panel, resize to 400px
- [ ] Close app, reopen → verify Properties at 400px
- [ ] Use File → Open → verify file dialog appears
- [ ] Open file → verify added to Recent Files
- [ ] Test all menu keyboard shortcuts (Ctrl+S, etc.)

**Success Criteria**: Minimum 21/25 tests passing (84%)

---

### Phase 3: Settings Dialog (Week 4)
**Deliverables**:
- [ ] Settings dialog framework
- [ ] Category tree (General, Editor, Appearance)
- [ ] Link to ConfigManager Pydantic models
- [ ] Search/filter settings
- [ ] Reset to defaults button

**Testing Requirements** (Target: 85% pass rate):

#### Unit Tests (`tests/test_settings_model.py`)
- [ ] Test category tree construction
- [ ] Test setting value get/set
- [ ] Test setting validation (min/max)
- [ ] Test setting type conversion
- [ ] Test reset to default value

#### Unit Tests (`tests/test_settings_search.py`)
- [ ] Test search by keyword ("theme" finds General.theme)
- [ ] Test fuzzy search ("fnt" finds "font")
- [ ] Test search result highlighting
- [ ] Test empty search returns all settings

#### Unit Tests (`tests/test_settings_widgets.py`)
- [ ] Test checkbox widget → bool setting
- [ ] Test spinbox widget → int setting
- [ ] Test lineedit widget → str setting
- [ ] Test combobox widget → enum setting
- [ ] Test color picker widget → color setting

#### Integration Tests (`tests/test_settings_dialog.py`)
- [ ] Test open dialog → displays current values
- [ ] Test change theme → ConfigManager updated
- [ ] Test change font size → emits on_changed event
- [ ] Test reset category → all defaults restored
- [ ] Test search "theme" → navigates to setting

#### Integration Tests (`tests/test_settings_persistence.py`)
- [ ] Test change 5 settings → save → close app
- [ ] Test reopen app → verify settings persisted
- [ ] Test ConfigManager reactivity (change → UI updates)
- [ ] Test invalid value → validation error shown

**Manual Testing Checklist**:
- [ ] Open Settings (Ctrl+,)
- [ ] Navigate to General → Appearance
- [ ] Change theme to "Dark"
- [ ] Close dialog → verify UI theme changes
- [ ] Reopen Settings → verify "Dark" selected
- [ ] Search for "font" → verify Editor.font appears
- [ ] Reset Editor category → verify default font

**Success Criteria**: Minimum 20/24 tests passing (83%)

---

### Phase 4: Polish & Commands (Week 5)
**Deliverables**:
- [ ] Command Palette (Ctrl+Shift+P)
- [ ] Keyboard shortcuts working for all menus
- [ ] Context menus for tabs (Close, Close Others, etc.)
- [ ] Drag to reorder tabs within split
- [ ] Comprehensive end-to-end testing

**Testing Requirements** (Target: 85% pass rate):

#### Unit Tests (`tests/test_command_palette.py`)
- [ ] Test command search (fuzzy matching)
- [ ] Test command execution from palette
- [ ] Test command history (recent first)
- [ ] Test keyboard navigation (up/down/enter)
- [ ] Test command filtering by category

#### Unit Tests (`tests/test_tab_context_menu.py`)
- [ ] Test "Close" action
- [ ] Test "Close Others" action
- [ ] Test "Close All" action
- [ ] Test "Close to Right" action
- [ ] Test menu enable/disable (can't close last tab)

#### Unit Tests (`tests/test_keyboard_shortcuts.py`)
- [ ] Test Ctrl+N → New Document
- [ ] Test Ctrl+O → Open Dialog
- [ ] Test Ctrl+S → Save Document
- [ ] Test Ctrl+W → Close Tab
- [ ] Test Ctrl+Tab → Next Document
- [ ] Test Ctrl+\ → Split Horizontal
- [ ] Test Ctrl+Shift+P → Command Palette

#### Integration Tests (`tests/test_tab_drag_drop.py`)
- [ ] Test drag tab within same split → reorder
- [ ] Test drag tab to different split → move
- [ ] Test drag tab to edge → create new split
- [ ] Test drag tab to empty area → no action

#### End-to-End Tests (`tests/test_e2e_workflow.py`)
- [ ] Test complete workflow: Open app → New doc → Edit → Split → Save → Close
- [ ] Test keyboard-only workflow (no mouse)
- [ ] Test 20 documents in 10 splits → verify performance
- [ ] Test all menus accessible via keyboard
- [ ] Test settings persist across restart

**Manual Testing Checklist**:
- [ ] Open Command Palette (Ctrl+Shift+P)
- [ ] Type "split" → verify split commands appear
- [ ] Execute "Split Horizontal" from palette
- [ ] Right-click tab → verify context menu
- [ ] Select "Close Others" → verify only current tab remains
- [ ] Drag tab to new position → verify reorder
- [ ] Use only keyboard for 5 minutes → verify all actions possible

**Success Criteria**: Minimum 21/25 tests passing (84%)

---

## 🧪 Overall Testing Strategy

### Test Coverage Requirements
**Target**: 85% pass rate per phase before moving to next phase

### Test Pyramid
```
        ╱ E2E Tests ╲      (10%)
       ╱─────────────╲
      ╱  Integration  ╲   (30%)
     ╱─────────────────╲
    ╱   Unit Tests     ╲  (60%)
   ╱───────────────────╲
```

### Running Tests
```bash
# From foundation template root
python -m pytest templates/foundation/tests/test_split_manager.py -v

# Run all GUI tests
python -m pytest templates/foundation/tests/test_*.py --cov=src/ui --cov-report=html

# Run specific phase tests
python -m pytest templates/foundation/tests/ -m "phase1" -v
```

### Test Organization
```
templates/foundation/tests/
├── conftest.py                    # Shared fixtures (qapp, mock services)
├── test_split_manager.py          # Phase 1
├── test_document_view.py          # Phase 1
├── test_layout_state.py           # Phase 1
├── test_split_integration.py      # Phase 1
├── test_dock_manager.py           # Phase 2
├── test_menu_builder.py           # Phase 2
├── test_action_registry.py        # Phase 2
├── test_panel_persistence.py      # Phase 2
├── test_settings_model.py         # Phase 3
├── test_settings_dialog.py        # Phase 3
├── test_command_palette.py        # Phase 4
├── test_keyboard_shortcuts.py     # Phase 4
└── test_e2e_workflow.py           # Phase 4
```

### CI/CD Integration
**Automated Testing**: All tests run on every commit
**Phase Gates**: Cannot merge to main unless 85% pass rate achieved
**Coverage Reports**: Generated and uploaded to artifacts

### Performance Benchmarks
- Layout save/restore: < 100ms
- Open 20 documents: < 2 seconds
- Split/merge operation: < 50ms
- Settings dialog open: < 200ms

---

## 📚 Documentation

### User Guides
- **Working with Splits**: How to arrange documents
- **Customizing Layout**: Saving layout presets
- **Keyboard Shortcuts**: Complete reference

### Developer Guides
- **Creating Custom Document Types**: Extend `DocumentView`
- **Adding Custom Panels**: Inherit from `BasePanelWidget`
- **Registering Menu Actions**: Use `CommandBus`

---

## 🚀 Success Criteria

✅ Users can split documents in any configuration (side/top/nested)
✅ Panel positions persist across app restarts
✅ Settings dialog covers all config options
✅ All commands accessible via menu or keyboard
✅ Layout state saves/restores in <100ms
✅ App supports 20+ open documents with 5+ splits without lag

---

## 🔮 Future Enhancements (Deferred)

### File Explorer with Custom Handling
**When**: After basic document management is stable
**Features**:
- Multi-root workspace
- Custom file type handlers (not Python's default open)
- Thumbnail previews
- File watching and auto-refresh

### Advanced Panels
**When**: On-demand
- Search panel (find in files with regex)
- Outline panel (document structure)
- Timeline panel (file history/git)

### Themes
**When**: After core features working
- Dark/Light theme system
- Custom QSS stylesheets

#### 4.1 Multi-Root Workspace Explorer
**Inspiration**: VS Code Explorer + ACDSee Folder Tree + Directory Opus
**Features**:
- **Multiple root folders** displayed simultaneously
- **Tree view** with expand/collapse
- **File icons** based on type (extension-aware)
- **Inline rename** with F2
- **Drag & drop** to move/copy files
- **Breadcrumb navigation** at top
- **Filter by name/extension**
- **Quick preview** panel (image/text)
- **Right-click context menu** (New, Delete, Reveal in OS)

**Integration with Template**:
- File operations dispatched via `CommandBus`
- File watching using `QFileSystemWatcher` or `watchdog`
- Thumbnails cached via `AssetManager`
- File operations logged to `JournalService`

**Implementation Classes**:
```
src/ui/explorer/
├── explorer_panel.py         # Main panel widget
├── file_tree_model.py        # QAbstractItemModel for file system
├── file_tree_view.py         # Custom QTreeView
├── breadcrumb_widget.py      # Path navigation
├── preview_panel.py          # Quick preview (image/text)
└── file_operations.py        # Copy/Move/Delete commands
```

**Advanced Features** (Directory Opus inspired):
- **Dual-pane mode** - Two explorers side-by-side
- **Flat view** - Show all nested files in one list
- **Column customization** - Size, Modified Date, Type
- **Archive browsing** - Navigate into ZIP/RAR as folders

---

### 5. Panel Context Management

#### 5.1 Smart Panel System
**Features**:
- **Panel registry** - Add custom panels dynamically
- **Context awareness** - Panels react to active document
- **Panel linking** - Synchronize state between panels
- **Panel groups** - Auto-show/hide related panels
- **Panel shortcuts** - Keybindings to toggle visibility

**Common Panels**:
- **Explorer** - File browser (see §4)
- **Outline** - Document structure (headers, symbols)
- **Search** - Find in files with regex support
- **Properties** - Inspector for selected object
- **Timeline** - File history / Git integration
- **Output** - Logs and console (see §6)
- **Tasks** - Background job monitor
- **Extensions** - Plugin marketplace

**Integration with Template**:
- Panels inherit from `BasePanelWidget` (extends `QDockWidget`)
- Panel ViewModels subscribe to `ServiceLocator.bus` events
- Panel state persisted in `ConfigManager`

**Implementation Classes**:
```
src/ui/panels/
├── base_panel.py             # Abstract panel with lifecycle
├── panel_registry.py         # Central panel manager
├── outline_panel.py          # Document outline
├── search_panel.py           # Find in files
└── tasks_panel.py            # Background job UI
```

---

### 6. Main Menu & Command Palette

#### 6.1 Extensible Menu System
**Features**:
- **Dynamic menus** - Items added by plugins
- **Recent files** submenu
- **Context-sensitive** - Menu items enable/disable based on state
- **Keyboard accelerators** displayed
- **Toolbar** with customizable buttons

#### 6.2 Command Palette (VS Code Ctrl+Shift+P)
**Features**:
- **Fuzzy search** for all commands
- **Command history** - recently used commands first
- **Keybinding display** next to command name
- **Quick file open** (Ctrl+P mode)

**Integration with Template**:
- Commands registered in `CommandBus`
- Command metadata includes name, description, shortcut
- Menu builder reads from command registry

**Implementation Classes**:
```
src/ui/menus/
├── menu_builder.py           # Dynamically build menus
├── command_palette.py        # Search dialog
├── action_registry.py        # QAction pool
└── keybinding_manager.py     # Shortcut customization
```

---

### 7. Theme & Styling System

#### 7.1 Complete Theme Support
**Features**:
- **Dark/Light themes** with automatic OS detection
- **Accent color** customization
- **Custom QSS stylesheets** per theme
- **Icon sets** (SVG) that adapt to theme
- **Live theme switching** without restart

**Integration with Template**:
- Theme stored in `ConfigManager.data.general.theme`
- Theme change triggers `ObserverEvent` for UI refresh
- Icon provider using `QIconEngine` for dynamic coloring

**Implementation Classes**:
```
src/ui/theming/
├── theme_manager.py          # Load/apply themes
├── themes/
│   ├── dark.qss
│   ├── light.qss
│   └── icons/                # SVG icon set
└── style_constants.py        # Color palette definitions
```

---

## 🔗 Integration with Core Template Systems

### Service Locator
- All UI managers registered as systems: `DockManager`, `DocumentManager`, `ThemeManager`
- Lifecycle managed via `initialize()` / `shutdown()`

### MVVM Pattern
- Each major UI component has a ViewModel
- ViewModels access core services via `ServiceLocator`
- ViewModels expose properties and commands for binding

### Command Bus
- All user actions are commands: `SaveDocumentCommand`, `CloseTabCommand`
- Commands can be invoked from menus, shortcuts, or command palette
- Undo/redo stack built on command history

### Configuration
- All UI settings stored in `ConfigManager`
- Settings dialog edits Pydantic models directly
- Changes propagate via `on_changed` events

### Task System
- Long-running operations (file indexing, thumbnail generation) run as tasks
- Progress displayed in Tasks panel
- Tasks can be cancelled by user

### Journal
- All file operations logged: "User opened file X", "User deleted folder Y"
- Searchable audit trail

---

## 📋 Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Refactor `MainWindow` to use `DockManager`
- [ ] Implement `DocumentManager` with tab support
- [ ] Create `BasePanelWidget` and panel registry
- [ ] Implement layout serialization

### Phase 2: Essential Panels (Week 3-4)
- [ ] File Explorer panel with tree view
- [ ] Output panel with log streaming
- [ ] Properties/Inspector panel
- [ ] Outline panel

### Phase 3: Settings & Theming (Week 5)
- [ ] Build settings dialog framework
- [ ] Implement theme manager
- [ ] Create dark/light themes
- [ ] Icon adaptation system

### Phase 4: Advanced Features (Week 6-7)
- [ ] Command palette with fuzzy search
- [ ] Multi-root workspace support
- [ ] Split editor views
- [ ] Dual-pane explorer mode

### Phase 5: Polish & Documentation (Week 8)
- [ ] Comprehensive testing
- [ ] User guide for UI customization
- [ ] Plugin API documentation
- [ ] Example custom panel/document type

---

## 🧪 Testing Strategy

### Unit Tests
- Panel lifecycle (show/hide/close)
- Layout serialization round-trip
- Command registration and execution
- Theme application

### Integration Tests
- Dock dragging and repositioning
- Document tab management
- File operations in explorer
- Settings persistence

### Manual Testing
- UI responsiveness with 100+ open tabs
- Layout restoration after restart
- Theme switching without artifacts
- Keyboard navigation

---

## 📚 Documentation Requirements

### User Documentation
- **Getting Started**: Creating first workspace, opening files
- **Customization Guide**: Creating custom themes, layouts
- **Keyboard Shortcuts**: Complete reference

### Developer Documentation
- **Creating Custom Panels**: API guide with example
- **Registering Document Types**: How to add new file viewers
- **Theme Development**: QSS reference and color variables
- **Command Registration**: Adding new menu items and shortcuts

---

## 🎨 Visual Mockups (Descriptions)

### Main Window Layout
```
┌─────────────────────────────────────────────────────────┐
│ File  Edit  View  Tools  Window  Help          [_][□][X]│
├─────────────────────────────────────────────────────────┤
│ Toolbar: [New] [Open] [Save] | [Undo] [Redo] | Search  │
├──────────┬──────────────────────────────┬───────────────┤
│ EXPLORER │  Document.txt    Image.png  ▼│  PROPERTIES   │
│ ▼ Project│  ┌──────────────────────────┐│  ╔══════════╗ │
│  📁 src  │  │ # Document Title         ││  ║ Selected ║ │
│   📄 main│  │                          ││  ║ Object   ║ │
│   📄 ui  │  │ Content here...          ││  ║ Info     ║ │
│  📁 docs │  │                          ││  ╚══════════╝ │
│  📁 tests│  └──────────────────────────┘│               │
├──────────┴──────────────────────────────┴───────────────┤
│ OUTPUT                                          ✕  ↓  ▲ │
│ [Info] Application started                              │
│ [Debug] Loaded 3 panels                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Success Criteria

### Functional Requirements
✅ Users can drag panels to any edge
✅ Users can save/load workspace layouts
✅ Settings dialog covers all config options
✅ File explorer supports multi-folder workspaces
✅ Themes switch without visual glitches
✅ Command palette finds commands by fuzzy name

### Performance Requirements
✅ UI remains responsive with 50+ open documents
✅ File tree loads 10,000 files in <2 seconds
✅ Theme switching completes in <500ms
✅ Tab switching is instant (<50ms)

### Usability Requirements
✅ Users can complete common tasks without mouse
✅ All actions have keyboard shortcuts
✅ Tooltips explain every UI element
✅ Drag-and-drop works intuitively

---

## 🔌 Plugin API

### Extensibility Points
Developers can extend the system by:
1. **Registering custom document types** via `DocumentFactory`
2. **Adding new panels** to `PanelRegistry`
3. **Defining new commands** in `CommandBus`
4. **Contributing settings** to categories
5. **Providing custom themes**

### Example: Custom Panel Registration
```python
from src.ui.panels import BasePanelWidget, panel_registry

class MyCustomPanel(BasePanelWidget):
    def __init__(self, locator):
        super().__init__("My Panel", locator)
        # Setup UI
    
    async def initialize(self):
        # Load data
        pass

# In plugin init:
panel_registry.register("my_panel", MyCustomPanel)
```

---

## 📝 Open Questions for Review

1. **MDI vs. Tabs**: Should we support true MDI (free-floating child windows) or stick with tabs?
2. **File Limits**: What's the max reasonable file count for the explorer tree?
3. **Touch Support**: Should panels be touch-friendly (larger hit targets)?
4. **Localization**: Should UI strings be i18n-ready from the start?
5. **Accessibility**: What level of screen reader support is needed?

---

**Next Steps**: Review this plan, discuss priorities, and begin Phase 1 implementation.
