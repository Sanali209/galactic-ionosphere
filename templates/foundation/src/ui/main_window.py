from PySide6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Slot
from loguru import logger

class MainWindow(QMainWindow):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle("Foundation App")
        self.resize(1200, 800)
        
        # 1. Central Widget
        self.central_widget = QLabel("Central Content Area")
        self.central_widget.setAlignment(Qt.AlignCenter)
        self.central_widget.setStyleSheet("font-size: 24px; color: gray;")
        self.setCentralWidget(self.central_widget)
        
        # MVVM Binding
        self.bridge.statusMessageChanged.connect(self.update_status)
        
    @Slot(str)
    def update_status(self, msg):
        self.central_widget.setText(f"Status: {msg}")

        
        # 2. Docks
        self.create_docks()
        
        # 3. Menu
        self.create_menu()
        
        logger.info("MainWindow initialized.")
        
    def create_docks(self):
        # Left Dock
        self.dock_left = QDockWidget("Explorer", self)
        self.dock_left.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.dock_left.setWidget(QLabel("Tree View Placeholder"))
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_left)
        
        # Right Dock
        self.dock_right = QDockWidget("Properties", self)
        self.dock_right.setAllowedAreas(Qt.RightDockWidgetArea)
        self.dock_right.setWidget(QLabel("Property Grid Placeholder"))
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_right)
        
        # Bottom Dock (Journal/Logs)
        self.dock_bottom = QDockWidget("Journal & Output", self)
        self.dock_bottom.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.dock_bottom.setWidget(QLabel("Log View Placeholder"))
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_bottom)

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Exit", self.close)
        
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.dock_left.toggleViewAction())
        view_menu.addAction(self.dock_right.toggleViewAction())
        view_menu.addAction(self.dock_bottom.toggleViewAction())
