from PySide6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from loguru import logger

from src.ui.widgets.gallery import GalleryWidget
from src.ui.widgets.sidebar import SidebarWidget
from src.ui.widgets.properties import PropertiesWidget
from src.ui.widgets.journal import JournalWidget

class MainWindow(QMainWindow):
    def __init__(self, bridge, fs_model, tag_model, grid_model, journal_model=None):
        super().__init__()
        self.setWindowTitle("Galactic Ionosphere (QtWidgets)")
        self.resize(1400, 900)
        
        self.bridge = bridge
        
        # 1. Central Widget (Gallery)
        self.gallery_widget = GalleryWidget(grid_model)
        self.setCentralWidget(self.gallery_widget)
        
        # 2. Left Dock (Sidebar)
        self.dock_solution = QDockWidget("Solution Explorer", self)
        self.dock_solution.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.sidebar_widget = SidebarWidget(bridge, fs_model, tag_model)
        self.dock_solution.setWidget(self.sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_solution)
        
        # 3. Right Dock (Properties)
        self.dock_props = QDockWidget("Properties", self)
        self.dock_props.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.props_widget = PropertiesWidget()
        self.dock_props.setWidget(self.props_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_props)
        
        # 4. Bottom Dock (Journal)
        self.dock_output = QDockWidget("Journal", self)
        self.dock_output.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.journal_widget = JournalWidget(journal_model)
        self.dock_output.setWidget(self.journal_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_output)
        
        # 5. Menu
        self.create_menu()
        
        # 6. Wiring Signals
        self._connect_signals()
        
        # Initial Refresh
        self.bridge.refreshGallery()
        
    def _connect_signals(self):
        # Sidebar -> Bridge
        self.sidebar_widget.folderSelected.connect(self.bridge.filterByFolder)
        self.sidebar_widget.tagSelected.connect(self.bridge.filterByTag)
        
        # Gallery -> Bridge
        self.gallery_widget.selectionChanged.connect(self.bridge.selectImage)
        
        # Bridge -> Properties
        self.bridge.imageSelected.connect(self.props_widget.set_data)
        
        # Properties -> Bridge
        self.props_widget.metadataChanged.connect(self.bridge.updateImageMetadata)

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        import_action = file_menu.addAction("Import Folder...")
        import_action.triggered.connect(self.open_import_dialog)
        
        wipe_action = file_menu.addAction("Wipe DB")
        wipe_action.triggered.connect(self.bridge.wipeDb)
        
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        view_menu = menubar.addMenu("&View")
        
        refresh_action = view_menu.addAction("Refresh")
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.bridge.refreshGallery)
        
        view_menu.addSeparator()
        view_menu.addAction(self.dock_solution.toggleViewAction())
        view_menu.addAction(self.dock_props.toggleViewAction())
        view_menu.addAction(self.dock_output.toggleViewAction())

    def open_import_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        if folder:
            self.bridge.importFolder(folder)

    def closeEvent(self, event):
        # self.write_settings() # Todo: implement settings persistence for widgets
        super().closeEvent(event)
