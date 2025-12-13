# Gap Analysis: Gallery Desktop

This document outlines the discrepancies between the initial "Gallery Desktop" project plan and the current codebase implementation. It serves as a guide for upcoming development phases to bridge these gaps.

## 1. Tech Stack

| Component | Plan Requirement | Current Status | Action |
| :--- | :--- | :--- | :--- |
| **Database Driver** | "New pymongo async client - replace motor" | **Migrated** to `pymongo` AsyncMongoClient. | **Done** |
| **Metadata** | `pyexiv2` | Verified installed. | **Done** |
| **Vector DB** | `qdrant` | Added `qdrant-client` to `requirements.txt`. | **Done** |
| **UI Framework** | PySide6 + QML | PySide6 is used. | Maintain. |

## 2. UI Components

The plan describes a rich, IDE-like interface (like Visual Studio). The current implementation is a mix of complete panels and simplified views.

*   **Directory View (Tree)**:
    *   *Plan*: Hierarchical tree of folders on physical disk, reactive updates.
    *   *Current*: **Partially Implemented**. A `FileExplorer.qml` exists using `FolderListModel` (Flat List), not a recursive Tree View. It allows navigation but lacks the hierarchical "Overview" of a tree.
*   **Tag View (Tree)**:
    *   *Plan*: Hierarchical tree of database tags, reactive updates.
    *   *Current*: **Partially Implemented**. `TagTreeModel` (Python) implements the hierarchy logic, but the UI (`SidebarPanel.qml`) uses a simple `ListView`, flattening the visualization.
*   **Search Panel**:
    *   *Plan*: Advanced mode with tree view for query building (AND/OR/NOT logic).
    *   *Current*: **Missing**. Only a basic text-based `SearchBox` exists.
*   **Centralized Settings Window**:
    *   *Plan*: "Like Visual Studio", left side vertical tabs, pages for settings.
    *   *Current*: **Implemented**. `SettingsPanel.qml` follows the requested design (Vertical tabs, StackLayout pages) and binds to config.
*   **Image Grid View**:
    *   *Plan*: Virtualization for large amounts of images, reactive.
    *   *Current*: **Implemented**. `GalleryView.qml` uses `GridView` which supports UI virtualization by default. Performance on 10k+ items needs verification.

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
*   **Import Pipeline**:
    *   *Plan*: "Choose folder and immediate process files in folder and subfolders".
    *   *Current*: **Implemented**. `backendBridge.importFolder` (and `_do_import_folder`) performs a recursive walk and processing.

## 4. Data Entities

*   **Polymorphism**:
    *   *Plan*: `galleryEntities` stores images, 3d models, faces.
    *   *Current*: Implemented via `BaseEntity` and `_cls`.
*   **Relations**:
    *   *Plan*: Universal relation system.
    *   *Current*: `Reference` table exists. UI for managing these is **Missing**.

## Recommendations

1.  **Refine UI Views**: The `FileExplorer` and `TagView` need to be upgraded from Lists to Trees (`TreeView` in QML is complex; consider `QTreeView` widget wrapped or `TreeView` from QtQuick Controls 1/Lab).
2.  **Dependency Fix**: Add `pyexiv2` and `qdrant-client` immediately.
3.  **Migration**: Schedule the Motor -> PyMongo Async migration early.
