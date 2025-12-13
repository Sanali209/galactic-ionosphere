from PySide6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtCore import QUrl, Qt, QSize
from PySide6.QtQml import QQmlContext

import os
import sys


class MainWindow(QMainWindow):
    def __init__(self, bridge, fs_model, tag_model, grid_model, journal_model=None):
        super().__init__()
        self.setWindowTitle("Galactic Ionosphere (Hybrid)")
        self.resize(1400, 900)
        
        # Store models first, as bridge might need them
        self.fs_model = fs_model
        self.tag_model = tag_model
        self.grid_model = grid_model
        self.journal_model = journal_model # Store it
        # The following line is an attempt to instantiate BackendBridge within MainWindow,
        # but the MainWindow's __init__ already receives 'bridge' as an argument.
        # If BackendBridge is to be instantiated here, 'bridge' should not be an argument
        # to MainWindow, and 'importer' and 'search_service' would need to be defined.
        # Assuming the intent was to ensure tag_model is passed to the existing bridge object
        # if it were instantiated here, but since 'bridge' is passed in, this line is
        # likely a misunderstanding of the current architecture.
        # For now, I will assume the user wants to keep the passed-in bridge.
        # If the intent is to instantiate BackendBridge here, the MainWindow signature
        # and the origin of 'importer' and 'search_service' would need to be clarified.
        # self.bridge = BackendBridge(self.importer, self.search_service, self.grid_model, self.journal_model, self.fs_model, self.tag_model) # Store it
        self.bridge = bridge
        
        # 1. Central Widget (Gallery / Document Area)
        self.central_widget = QQuickWidget()
        self.central_widget.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.central_widget, "components/TabManager.qml") # Using TabManager as center for now
        self.setCentralWidget(self.central_widget)
        
        # 2. Left Dock (Solution Explorer)
        self.dock_solution = QDockWidget("Solution Explorer", self)
        self.dock_solution.setObjectName("SolutionExplorerDock")
        self.dock_solution.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.qml_solution = QQuickWidget()
        self.qml_solution.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_solution, "panels/SidebarPanel.qml")
        
        self.dock_solution.setWidget(self.qml_solution)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_solution)
        
        # 3. Right Dock (Properties)
        self.dock_props = QDockWidget("Properties", self)
        self.dock_props.setObjectName("PropertiesDock")
        self.dock_props.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.qml_props = QQuickWidget()
        self.qml_props.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_props, "panels/PropertiesPanel.qml")
        
        self.dock_props.setWidget(self.qml_props)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_props)
        
        # 4. Bottom Dock (Journal / Output)
        self.dock_output = QDockWidget("Journal", self)
        self.dock_output.setObjectName("JournalDock")
        self.dock_output.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        
        self.qml_output = QQuickWidget()
        self.qml_output.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_output, "panels/JournalPanel.qml") # Use new panel
        
        self.dock_output.setWidget(self.qml_output)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_output)

        # 6. Settings Dialog (Hidden)
        from PySide6.QtWidgets import QDialog, QVBoxLayout
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Settings")
        self.settings_dialog.resize(800, 600)
        self.settings_dialog_layout = QVBoxLayout(self.settings_dialog)
        self.settings_dialog_layout.setContentsMargins(0,0,0,0)
        
        self.qml_settings = QQuickWidget()
        self.qml_settings.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_settings, "panels/SettingsPanel.qml")
        self.settings_dialog_layout.addWidget(self.qml_settings)


        # 7. Initial Data Load & Persistence
        self.read_settings()

        # 8. Menu Bar (Must be after actions/dialogs are created)
        self.create_menu()
        
        # We use a slight delay or just call it, assuming loop is running.
        self.bridge.refreshGallery()
        
    def closeEvent(self, event):
        self.write_settings()
        super().closeEvent(event)
        
    def read_settings(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("Galactic", "Ionosphere")
        geometry = settings.value("geometry")
        window_state = settings.value("windowState")
        
        if geometry:
            self.restoreGeometry(geometry)
        if window_state:
            self.restoreState(window_state)
            
    def write_settings(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("Galactic", "Ionosphere")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def create_menu(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        import_action = file_menu.addAction("Import Folder...")
        import_action.triggered.connect(self.open_import_dialog)
        
        vector_action = file_menu.addAction("Generate Missing Vectors (AI)")
        vector_action.triggered.connect(lambda: self.bridge.vectorizeAll())
        
        file_menu.addSeparator()
        
        # Settings Menu Item
        settings_action = file_menu.addAction("Settings")
        settings_action.triggered.connect(self.settings_dialog.show)
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = menubar.addMenu("&View")
        
        # Toggle Actions for Docks
        view_menu.addAction(self.dock_solution.toggleViewAction())
        view_menu.addAction(self.dock_props.toggleViewAction())
        view_menu.addAction(self.dock_output.toggleViewAction())

    def open_import_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        if folder:
            self.bridge.importFolder(folder)

    def setup_qml_widget(self, widget, qml_rel_path):
        """Helper to inject context and load source."""
        ctx = widget.rootContext()
        ctx.setContextProperty("backendBridge", self.bridge)
        ctx.setContextProperty("fileSystemModel", self.fs_model)
        ctx.setContextProperty("tagModel", self.tag_model)
        ctx.setContextProperty("galleryModel", self.grid_model)
        ctx.setContextProperty("journalModel", self.journal_model)
        
        base_path = os.path.dirname(__file__)
        qml_path = os.path.join(base_path, "qml", qml_rel_path)
        widget.setSource(QUrl.fromLocalFile(qml_path))
