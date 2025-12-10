# Phased Implementation Roadmap

**Goal**: Build the "Local Gallery & AI Manager" desktop application.
**Status**: Phases 1-3 Complete. Phases 4-6 Partially Complete (Refactoring Required).

## Phase 1: Domain Modeling & Data Layer (✅ Complete)
- [x] **1.1. Entity Models**: `ImageRecord`, `Tag`, `Detection`.
- [x] **1.2. Relation System**: `Reference` table and Manager.
- [x] **1.3. File Handling**: `FileHandler` base implementations.
- [x] **Verification**: Unit and Integration tests passing.

## Phase 2: The Engine (Pipelines & Processing) (✅ Complete)
- [x] **2.1. Task System**: Persistent Task Records & Dispatcher.
- [x] **2.2. Import Pipeline**: `FileMonitor`, `MetadataService` (Stubbed), `ThumbnailService`.
- [x] **2.3. Processing Logic**: Full Import -> Hash -> Meta -> Thumb flow.
- [x] **Verification**: Import flow verified.

## Phase 3: The Brain (AI & Vectors) (✅ Complete)
- [x] **3.1. Vector Capabilities**: Qdrant Driver (needs dep), CLIP Embeddings.
- [x] **3.2. AI Workers**: Vector Generation background tasks.
- [x] **3.3. Search Engine**: Semantic Search (Text-to-Image).
- [x] **Verification**: Search relevance verified.

## Phase 4: Technical Debt & Gap Remediation (🔄 In Progress)
*Focus: Aligning codebase with original plan and modernizing stack.*

- [ ] **4.1. Dependency Updates**:
    -   Replace `motor` with `pymongo` (AsyncMongoClient) as `motor` is deprecated.
    -   Add `qdrant-client` to requirements.
    -   Add `pyexiv2` and implement real metadata extraction.
- [ ] **4.2. System Gaps**:
    -   Implement "Folder Scan" logic (recursive walk for existing files).
    -   Implement XMP Write-back capability.
    -   Implement Mongo-based Journal/Log system.

## Phase 5: User Interface Construction (🚧 Partially Complete)
*Focus: Building the IDE-like "Visual Studio" interface defined in the plan.*

- [x] **5.1. Main Layout**: Sidebar, Grid, Inspector (Basic).
- [ ] **5.2. Directory Tree View**:
    -   Hierarchical file system view.
    -   Reactive updates from `FileMonitor`.
- [ ] **5.3. Tag Tree View**:
    -   Hierarchical view of `Tag` entities.
    -   Drag-and-drop support for tagging.
- [ ] **5.4. Advanced Search Panel**:
    -   Tree-based query builder (AND/OR groups).
- [ ] **5.5. Centralized Settings Window**:
    -   Vertical tabs (Visual Studio style).
    -   UI for `config.yaml` modification.

## Phase 6: Advanced UI & Interaction (🚧 Partially Complete)
- [x] **6.1. Docking System**: `DockPanel`, `LayoutManager`.
- [ ] **6.2. Relation UI**:
    -   Visual tool to view/edit `Reference` links between entities.
- [ ] **6.3. Image Grid Virtualization**:
    -   Ensure grid handles 10k+ images smoothly.

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
