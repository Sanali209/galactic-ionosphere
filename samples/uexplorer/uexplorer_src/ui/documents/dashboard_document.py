from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import uuid
from loguru import logger

from uexplorer_src.viewmodels.dashboard_viewmodel import DashboardViewModel
from uexplorer_src.ui.widgets.dashboard_card_widget import DashboardCardWidget
from src.ui.cardview.flow_layout import FlowLayout

class DashboardDocument(QWidget):
    """
    Dashboard main view container.
    """
    
    def __init__(self, locator, parent=None):
        super().__init__(parent)
        self.locator = locator
        self.id = f"dashboard_{uuid.uuid4().hex[:8]}"
        self._title = "Dashboard"
        
        # ViewModel
        self._viewmodel = DashboardViewModel(self.id, locator)
        
        # UI
        self._setup_ui()
        self._connect_signals()
        
        logger.debug("DashboardDocument initialized")
        
    def _setup_ui(self):
        """Build UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #f5f5f5; border-bottom: 1px solid #ddd;")
        toolbar.setFixedHeight(40)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 5, 10, 5)
        
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._viewmodel.refresh)
        tb_layout.addWidget(self._refresh_btn)
        tb_layout.addStretch()
        
        layout.addWidget(toolbar)
        
        # Scroll Area for FlowLayout
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setStyleSheet("background-color: #ffffff;")
        
        # Container Widget
        self._container = QWidget()
        self._container.setStyleSheet("background-color: transparent;")
        
        # Flow Layout
        self._flow_layout = FlowLayout(self._container, margin=16, h_spacing=16, v_spacing=16)
        
        self._scroll_area.setWidget(self._container)
        layout.addWidget(self._scroll_area)
        
    def _connect_signals(self):
        """Connect ViewModel signals."""
        self._viewmodel.items_changed.connect(self._on_items_changed)
    
    def _on_items_changed(self, items):
        """Update dashboard items."""
        # Clear existing items
        self._flow_layout.clear()
        
        # Add new items
        for item in items:
            widget = DashboardCardWidget(self._container)
            # Manually bind data since we aren't using CardView factory
            widget.bind_data(item)
            
            # Set fixed size for flow layout consistency
            # Longer cards for stats, standard for others
            widget.setFixedSize(260, 140)
            
            # Connect click signal
            widget.clicked.connect(self._on_item_clicked)
            
            self._flow_layout.addWidget(widget)
            
    def _on_item_clicked(self, item_id: str):
        """Handle card clicks."""
        # Find item in viewmodel (since we don't have a lookup map handy here, 
        # normally we'd keep a map, but asking viewmodel is safer if it had a get_item)
        # For now, we trust the ID passed back
        self._viewmodel.trigger_action(item_id)
            
    # Document Interface
    @property
    def title(self) -> str:
        return self._title
        
    def can_close(self) -> bool:
        return True
