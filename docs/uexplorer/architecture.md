# UExplorer Architecture

UExplorer is a reference application built on top of USCore, implementing a professional file manager interface.

## Core Patterns

### MVVM (Model-View-ViewModel)
-   **View**: `MainWindow`, `FileBrowserDocument`, `SearchPanel`.
-   **ViewModel**: `MainViewModel`, `BrowseViewModel`, `SearchPipeline`.
-   **Model**: `FileRecord`, `DirectoryRecord` (from UCoreFS).

All ViewModels inherit from `BindableBase` (`src.ui.mvvm.base`), enabling reactive UI updates via `BindableProperty`.

### Docking System
UExplorer uses **PySide6-QtAds** (Advanced Docking System) managed by `DockingService`.
-   **Documents**: Central tabs (File Browsers, Dashboard).
-   **DockWidgets**: Tool panels (Tags, Albums, Properties).
-   **Layouts**: Layouts are persisted to `session.json` and restored on startup.

### Managers
To prevent `MainWindow` from becoming a "God Object", logic is split into managers (`uexplorer_src.ui.managers`):
-   **DocumentManager**: Manages central tabs (browsers).
-   **FilterManager**: Handles global filtering logic.
-   **SelectionManager**: syncs selection across panes.
-   **SessionManager**: Persists window state, layout, and open tabs.

## Directory Structure

-   **`main.py`**: Application entry point.
-   **`uexplorer_src/`**: Source code.
    -   **`ui/`**: Widget implementations.
    -   **`viewmodels/`**: Business logic.
    -   **`commands/`**: Async command handlers.
