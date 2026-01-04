"""
General Settings Page

Application appearance and startup configuration.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QSpinBox, QComboBox
)


class GeneralSettingsPage(QWidget):
    """General application settings."""
    
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        appearance_layout.addRow("Theme:", self.theme_combo)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(12)
        appearance_layout.addRow("Font Size:", self.font_size_spin)
        
        layout.addWidget(appearance_group)
        
        # Startup
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        
        self.restore_layout = QCheckBox("Restore window layout")
        self.restore_layout.setChecked(True)
        startup_layout.addWidget(self.restore_layout)
        
        self.restore_last_dir = QCheckBox("Restore last directory")
        self.restore_last_dir.setChecked(True)
        startup_layout.addWidget(self.restore_last_dir)
        
        self.check_updates = QCheckBox("Check for updates on startup")
        startup_layout.addWidget(self.check_updates)
        
        layout.addWidget(startup_group)
        layout.addStretch()
