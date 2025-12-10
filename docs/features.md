# Project Features

## 🧠 Brain (AI & Intelligence)
-   **Semantic Search**: Find images using natural language queries (e.g., "cat in the snow").
-   **Vector Database**: High-performance vector storage using **Qdrant**.
-   **Modular AI**:
    -   **CLIP Integration**: Uses OpenAI's CLIP (via `transformers`) for image embedding.
    -   **Configurable Device**: Supports CPU and CUDA (if available).
    -   **Configurable Limits**: Adjustable result limits via Settings.

## 🖼️ Gallery & Core
-   **Import System**:
    -   Recursive folder scanning.
    -   Non-blocking background processing.
    -   Automatic duplicate detection (hash-based).
-   **Database**:
    -   **MongoDB** for metadata storage (fast, flexible).
    -   Reactive Document-Object Mapping via Motor.
-   **Performance**:
    -   Virtual scrolling for large galleries.
    -   Async I/O for all heavy operations.

## 🖥️ UI & Framework
-   **Hybrid Architecture**: PySide6 (Windowing) + QML (Rendering).
-   **Advanced Docking**:
    -   Draggable/Dockable panels (Solution Explorer, Properties, Output).
    -   Persists layout state.
-   **Centralized Settings**:
    -   Visual Studio-style settings window.
    -   JSON-based persistence (`config.json`).
    -   Real-time configuration updates.
-   **Theme**: Dark mode optimized for content visibility.
