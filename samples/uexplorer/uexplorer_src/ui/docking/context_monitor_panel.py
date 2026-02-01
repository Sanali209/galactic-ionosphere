"""
ContextMonitorPanel - Diagnostic tool for reactive sync.
"""
from typing import Optional
from PySide6.QtWidgets import QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, QHeaderView, QLabel
from PySide6.QtCore import Qt, QTimer
from uexplorer_src.ui.docking.panel_base import PanelBase
from src.ui.mvvm.sync_manager import ContextSyncManager
from loguru import logger

class ContextMonitorPanel(PanelBase):
    """
    Diagnostic panel to monitor reactive sync channels.
    Provides visibility into global state and subscriber hierarchy.
    """
    
    def setup_ui(self):
        """Build the diagnostic UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with status
        self._status_label = QLabel("Sync Status: 🟢 Active")
        self._status_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 4px;")
        layout.addWidget(self._status_label)
        
        # Tree display
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Channel / Subscriber", "Last Value / Property"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tree.setIndentation(15)
        self._tree.setStyleSheet("""
            QTreeWidget { background-color: #1e1e1e; color: #ccc; border: 1px solid #333; }
            QTreeWidget::item { padding: 4px; border-bottom: 1px solid #2a2a2a; }
            QTreeWidget::item:selected { background-color: #094771; }
        """)
        
        layout.addWidget(self._tree)
        
        # Toolbar
        self._btn_refresh = QPushButton("🔄 Manual Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        layout.addWidget(self._btn_refresh)
        
        # Auto-refresh timer
        self._timer = QTimer(self)
        self._timer.setInterval(2000) # 2 seconds
        self._timer.timeout.connect(self.refresh)
        
    def on_show(self):
        """Start auto-refresh when panel is visible."""
        self._timer.start()
        self.refresh()
        
    def on_hide(self):
        """Stop auto-refresh when panel is hidden."""
        self._timer.stop()
        
    def refresh(self):
        """Update the monitor tree with live registry data."""
        try:
            sync_manager = self.locator.get_system(ContextSyncManager)
        except (KeyError, AttributeError):
            # Fallback if not registered as system but exists as singleton/attribute
            # This handles cases where implementation might vary across branches
            from src.ui.mvvm.sync_manager import ContextSyncManager
            # We assume it was somehow initialized
            self._status_label.setText("Sync Status: 🔴 Manager Not Found")
            return
            
        if not sync_manager:
            self._status_label.setText("Sync Status: 🔴 No Manager")
            return

        self._status_label.setText("Sync Status: 🟢 Active")
        
        # Store expansion state to restore it after clear
        expanded_channels = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.isExpanded():
                expanded_channels.append(item.text(0))
        
        self._tree.clear()
        
        # Inspect sync_manager._registry
        # Note: _registry access is for diagnostic purposes
        for channel, registrations in sync_manager._registry.items():
            # Get current value from first active subscriber
            current_value = "N/A"
            for vm_ref, property_name in registrations:
                vm = vm_ref()
                if vm:
                    try:
                        val = getattr(vm, property_name, None)
                        current_value = str(val)[:60].replace("\n", " ") # Sanitize
                    except Exception:
                        current_value = "[Error retrieving value]"
                    break
            
            # Channel item
            channel_item = QTreeWidgetItem(self._tree)
            channel_item.setText(0, f"📡 {channel}")
            channel_item.setText(1, current_value)
            channel_item.setToolTip(1, str(current_value))
            channel_item.setData(0, Qt.UserRole, channel)
            
            bold_font = channel_item.font(0)
            bold_font.setBold(True)
            channel_item.setFont(0, bold_font)
            channel_item.setForeground(0, Qt.cyan)
            
            # Restore expansion
            if f"📡 {channel}" in expanded_channels:
                channel_item.setExpanded(True)
            
            # Subscriber items
            for vm_ref, property_name in registrations:
                vm = vm_ref()
                sub_item = QTreeWidgetItem(channel_item)
                if vm:
                    sub_item.setText(0, f"👤 {vm.__class__.__name__}")
                    sub_item.setText(1, f"🔗 {property_name}")
                    sub_item.setForeground(1, Qt.gray)
                else:
                    sub_item.setText(0, "💀 Dead Reference")
                    sub_item.setForeground(0, Qt.red)
