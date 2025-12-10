# Design Document: Local Gallery & AI Manager

## 1. Executive Summary
A high-performance desktop application for viewing, analyzing, and organizing large image collections. It combines the speed of **PySide6/QML** with the power of **AI** (LLMs, Vector Search) and a flexible **MongoDB** backend. Inspired by Adobe Photoshop Elements but focused on AI-driven organization.

## 2. Technology Stack

-   **Language**: Python 3.11+
-   **UI Framework**: PySide6 + QML (Modern, GPU-accelerated interface)
-   **Database**:
    -   **MongoDB** (via `motor`): Metadata, Settings, Logs, Relations.
    -   **Qdrant**: Vector database for Semantic Search (CLIP/BLIP embeddings).
-   **Core Libraries**:
    -   `pydantic`: Data validation and settings.
    -   `loguru`: Logging.
    -   `pyexiv2`: XMP/Metadata R/W.
    -   `Pillow` / `OpenCV`: Image processing.
    -   `watchdog`: File system monitoring.

## 3. System Architecture

### 3.1. Foundation Layer (Existing)
-   **Service Locator (`sl`)**: Central registry.
-   **Event Bus**: Decoupled communication.
-   **Reactive Config**: Settings management.
-   **Mongo ORM**: Async data access.

### 3.2. Core Systems
1.  **Import Pipeline**:
    -   Watches folders recursively.
    -   Extracts metadata (Dimensions, Size, XMP).
    -   Computes **Content Hash** (Pixel-based, not file-based).
    -   Generates Thumbnails.
2.  **Background Processing**:
    -   **Task Queue** (MongoDB-backed): Persistent tasks (e.g., "Generate Vectors", "Detect Objects").
    -   **AI Worker**: Consumes tasks to run heavyweight models (CLIP, DINO, LLMs) without freezing UI.
3.  **Thumbnail Cache**:
    -   Disk-based storage.
    -   Sharding: `cache/{xx}/{yy}/{hash}.jpg` (First 2 chars of hash).
    -   Backend: `DiskCache` or custom file manager.
4.  **File Data Handler**:
    -   Strategy pattern to handle different extensions (`.jpg`, `.png`, `.mp4`, `.obj`).

### 3.3. AI & Search
-   **Vector Search**: Hybrid search (Text + Vector).
-   **Object Detection**: Grounding DINO / YOLO for finding generic objects ("car", "face").
-   **Auto-Relation**: Background job to find duplicates or similar images via vector distance.

## 4. Data Model (Schema)

### 4.1. Entities (`galleryEntities`)
Single polymorphic collection for all manageable assets.

-   **`BaseEntity`**:
    -   `_id`: ObjectId
    -   `payload`: Dict (Confidence score, transformation data)

### 4.3. Tags (`tags`)
Hierarchical tag structure. Uses `parent_id` or Materialized Path for efficient tree retrieval.

-   **`Tag`**:
    -   `name`: String
    -   `parent_path`: String (e.g., "People|Family")

## 5. UI / UX (QML)

### 5.1. Main Layout
-   **Left Panel**:
    -   **Folder Tree**: Reactive file system view.
    -   **Tag Tree**: Database tag hierarchy.
    -   **Selectors**: Filter by Rating, Label.
-   **Center**:
    -   **Image Grid**: Virtualized GridView.
    -   **Cards**: Thumbnail, Badges (Rating, Tag count), Selection state.
-   **Right Panel (Toolbox)**:
    -   **Properties**: Metadata view.
    -   **Relations**: List of related items (Similars, Duplicates).
    -   **Search**:
        -   *Simple*: Text box.
        -   *Advanced*: Boolean query builder (AND/OR/NOT).

### 5.2. Features
-   **Settings Window**: Visual Studio style (Vertical tabs).
-   **Drag & Drop**: Tagging by dragging tags onto images.
-   **Reactive Updates**: UI listens to `on_change` events from Core.

## 6. Optimization Strategy
-   **Virtualization**: QML `GridView` loads only visible items.
-   **Lazy Loading**: Tree views load children on expand.
-   **Thumbnail Generation**: Fast, async, cached.
