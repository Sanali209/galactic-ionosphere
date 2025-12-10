from PySide6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtCore import QUrl, Qt, QSize
from PySide6.QtQml import QQmlContext

import os
import sys

class MainWindow(QMainWindow):
    def __init__(self, bridge, folder_model, tag_model, grid_model):
        super().__init__()
        self.setWindowTitle("Galactic Ionosphere (Hybrid)")
        self.resize(1400, 900)
        
        self.bridge = bridge
        self.folder_model = folder_model
        self.tag_model = tag_model
        self.grid_model = grid_model
        
        # 1. Central Widget (Gallery / Document Area)
        self.central_widget = QQuickWidget()
        self.central_widget.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.central_widget, "components/TabManager.qml") # Using TabManager as center for now
        self.setCentralWidget(self.central_widget)
        
        # 2. Left Dock (Solution Explorer)
        self.dock_solution = QDockWidget("Solution Explorer", self)
        self.dock_solution.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.qml_solution = QQuickWidget()
        self.qml_solution.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_solution, "panels/SidebarPanel.qml")
        
        self.dock_solution.setWidget(self.qml_solution)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_solution)
        
        # 3. Right Dock (Properties)
        self.dock_props = QDockWidget("Properties", self)
        self.dock_props.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.qml_props = QQuickWidget()
        self.qml_props.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_props, "panels/PropertiesPanel.qml")
        
        self.dock_props.setWidget(self.qml_props)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_props)
        
        # 4. Bottom Dock (Output)
        self.dock_output = QDockWidget("Output", self)
        self.dock_output.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        
        self.qml_output = QQuickWidget()
        self.qml_output.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.setup_qml_widget(self.qml_output, "panels/OutputPanel.qml")
        
        self.dock_output.setWidget(self.qml_output)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_output)

        # 5. Menu Bar
        self.create_menu()

        # 6. Initial Data Load
        # We use a slight delay or just call it, assuming loop is running.
        # Since main.py loop is robust, we can just call the slot which schedules task.
        self.bridge.refreshGallery()

    def create_menu(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        import_action = file_menu.addAction("Import Folder...")
        import_action.triggered.connect(self.open_import_dialog)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # View Menu
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
        ctx.setContextProperty("folderModel", self.folder_model)
        ctx.setContextProperty("tagModel", self.tag_model)
        ctx.setContextProperty("galleryModel", self.grid_model)
        
        base_path = os.path.dirname(__file__)
        qml_path = os.path.join(base_path, "qml", qml_rel_path)
        widget.setSource(QUrl.fromLocalFile(qml_path))
