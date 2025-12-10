# Phased Implementation Roadmap

**Goal**: Build the "Local Gallery & AI Manager" desktop application.
**Status**: Phases 1-6 Complete. Entering Phase 7 (Object Detection).

## Phase 1: Domain Modeling & Data Layer (✅ Complete)
- [x] **1.1. Entity Models**: `ImageRecord`, `Tag`, `Detection`.
- [x] **1.2. Relation System**: `Reference` table and Manager.
- [x] **1.3. File Handling**: `FileHandler` implementations.
- [x] **Verification**: Unit and Integration tests passing.

## Phase 2: The Engine (Pipelines & Processing) (✅ Complete)
- [x] **2.1. Task System**: Persistent Task Records & Dispatcher.
- [x] **2.2. Import Pipeline**: `FileMonitor`, `MetadataService`, `ThumbnailService`.
- [x] **2.3. Processing Logic**: Full Import -> Hash -> Meta -> Thumb flow.
- [x] **Verification**: Import flow verified.

## Phase 3: The Brain (AI & Vectors) (✅ Complete)
- [x] **3.1. Vector Capabilities**: Qdrant Driver, CLIP Embeddings.
- [x] **3.2. AI Workers**: Vector Generation background tasks.
- [x] **3.3. Search Engine**: Semantic Search (Text-to-Image).
- [x] **Verification**: Search relevance verified.

## Phase 4: UI Backend (The Bridge) (✅ Complete)
- [x] **4.1. Qt Integration**: `QApplication`, `BackendBridge` (Signals/Slots).
- [x] **4.2. View Models**: `GalleryGridModel`, `FolderTreeModel`.
- [x] **Verification**: Models correctly driving QML.

## Phase 5: User Interface (QML) (✅ Complete)
- [x] **5.1. Main Layout**: Sidebar, Grid, Inspector.
- [x] **5.2. Components**: `ImageCard`, `SearchBox`.
- [x] **5.3. Interactivity**: Selection, Scrolling.
- [x] **Verification**: Visual regression passed.

## Phase 6: Advanced UI (MDI & Docking) (✅ Complete)
- [x] **6.1. Docking System**: `DockPanel`, `LayoutManager`.
- [x] **6.2. Document Management**: `TabManager` (Central Area).
- [x] **6.3. Tool Windows**: Output Panel, Settings Panel (Visual Studio Style).
- [x] **6.4. Persistence**: `SettingsPanel` writes to `config.json`.
- [x] **Verification**: Docking and Settings persistence verified.

## Phase 7: Deep Intelligence (Object Detection) (Next)
**Objective**: Implement Granular Object Detection (YOLO/DINO).

- [ ] **7.1. Detection Models**:
    -   Define `Detection` schema in `ImageRecord`.
- [ ] **7.2. Global Detection Service**:
    -   Integrate `ultralytics` (YOLOv8).
    -   Create `DETECT_OBJECTS` task handler.
- [ ] **7.3. UI Visualization**:
    -   Overlay Bounding Boxes on `ImageCard`.
    -   "People/Objects" view in Sidebar.

## Phase 8: Professional UX (refinement)
- [ ] **8.1. Menu System**: Expand File/Edit/View menus.
- [ ] **8.2. Folder Navigation**: Drill-down File System navigation.
- [ ] **8.3. Window Management**: Save/Restore Layout state.
