# UI Architecture Overview

The UI is built using **PySide6 (Qt)** for the window management and **QML** for the internal content rendering, creating a high-performance, hybrid interface.

## 1. Main Window (`src.ui.main_window`)

The `MainWindow` manages the docking system and top-level menus. It does not render content directly but hosts `QQuickWidget` instances.

### Docking Areas
-   **Central Widget**: Main content area (Document Tabs / Gallery).
-   **Left Dock**: Solution Explorer / Sidebar.
-   **Right Dock**: Properties / Settings.
-   **Bottom Dock**: Output / Console.

### Settings Panel
A centralized configuration UI (`SettingsPanel.qml`) is available via `File > Settings`.
-   **Persistence**: Settings are automatically saved to `config.json`.
-   **Structure**: Uses a Visual Studio-like vertical tab layout.
-   **Context**: Changes are reactive and immediately affect the backend.

## 2. QML Integration (`src.ui.bridge`)

Communication between Python (Backend) and QML (Frontend) is handled by the `BackendBridge`.

### Key Components
-   **Signals**: `searchFinished(int)`, `imageSelected(...)`, `logMessage(str)`.
-   **Slots**: Methods callable from QML (e.g., `search()`, `importFolder()`).
-   **Generic Settings**: QML can read/write any config value using:
    -   `backendBridge.getSetting("section.key")`
    -   `backendBridge.setSetting("section.key", value)`

## 3. Theming
The UI uses a dark theme inspired by modern IDEs (VS Code), defined in QML with standard Qt Quick Controls 2 styling.
