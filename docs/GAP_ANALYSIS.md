# Gap Analysis: Gallery Desktop

This document outlines the discrepancies between the initial "Gallery Desktop" project plan and the current codebase implementation. It serves as a guide for upcoming development phases to bridge these gaps.

## 1. Tech Stack

| Component | Plan Requirement | Current Status | Action |
| :--- | :--- | :--- | :--- |
| **Database Driver** | "New pymongo async client - replace motor" | Uses `motor` (deprecated). | **Migrate** to `pymongo` (AsyncMongoClient). |
| **Metadata** | `pyexiv2` | Not installed; Stubs in `images.py`. | **Add Dependency** and implement wrappers. |
| **Vector DB** | `qdrant` | `vector_driver.py` exists but `qdrant-client` is missing from `requirements.txt`. | **Add Dependency** to `requirements.txt`. |
| **UI Framework** | PySide6 + QML | PySide6 is used. | Maintain. |

## 2. UI Components (Major Gaps)

The plan describes a rich, IDE-like interface (like Visual Studio). The current implementation is a basic viewer.

*   **Directory View (Tree)**:
    *   *Plan*: Hierarchical tree of folders on physical disk, reactive updates.
    *   *Current*: **Missing**. Only a flat `GalleryView` exists.
*   **Tag View (Tree)**:
    *   *Plan*: Hierarchical tree of database tags, reactive updates.
    *   *Current*: **Missing**.
*   **Search Panel**:
    *   *Plan*: Advanced mode with tree view for query building (AND/OR/NOT logic).
    *   *Current*: **Missing**. Basic search might exist in backend but no UI.
*   **Centralized Settings Window**:
    *   *Plan*: "Like Visual Studio", left side vertical tabs, pages for settings.
    *   *Current*: **Missing**. `config.json` exists, but no UI to edit it.
*   **Image Grid View**:
    *   *Plan*: Virtualization for large amounts of images, reactive.
    *   *Current*: `GalleryView.qml` exists but needs verification of virtualization capabilities.

## 3. Systems & Features

*   **Journal System**:
    *   *Plan*: Store journal/logs to MongoDB.
    *   *Current*: **Missing**. `loguru` logs to file/console. No Mongo integration found.
*   **Automatic Relation Finding**:
    *   *Plan*: Background process using vectors to find similars/duplicates and create relations.
    *   *Current*: **Missing**. Vector search exists, but the "Auto-Relation" logic is not implemented.
*   **XMP Write Back**:
    *   *Plan*: Possibility to write XMP data on user request.
    *   *Current*: **Missing**. Only extraction stubs exist.
*   **Initial Import Pipeline**:
    *   *Plan*: "Choose folder and immediate process files in folder and subfolders".
    *   *Current*: `importer.py` handles single file. `FileMonitor` handles new files. **Missing** "Scan Folder" command/logic to walk existing directories.

## 4. Data Entities

*   **Polymorphism**:
    *   *Plan*: `galleryEntities` stores images, 3d models, faces.
    *   *Current*: Implemented via `BaseEntity` and `_cls`.
*   **Relations**:
    *   *Plan*: Universal relation system.
    *   *Current*: `Reference` table exists. UI for managing these is **Missing**.

## Recommendations

1.  **Prioritize UI Construction**: The backend foundation is decent (aside from the Motor migration), but the UI is significantly behind the "Visual Studio-like" vision.
2.  **Dependency Fix**: Add `pyexiv2` and `qdrant-client` immediately.
3.  **Migration**: Schedule the Motor -> PyMongo Async migration early to avoid building more on a deprecated driver.
