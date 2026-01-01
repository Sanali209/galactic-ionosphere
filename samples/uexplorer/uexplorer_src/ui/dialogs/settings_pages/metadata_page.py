"""
Metadata Settings Page

Metadata extraction and mapping settings.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox, QLabel
)


class MetadataSettingsPage(QWidget):
    """Metadata extraction and mapping settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Automatic Processes
        auto_group = QGroupBox("Automatic Processes")
        auto_layout = QVBoxLayout(auto_group)
        
        self.auto_fill_blip = QCheckBox("Auto-fill description from AI caption (BLIP)")
        self.auto_fill_blip.setToolTip("Automatically copies generated AI captions to the file description if it's empty.")
        auto_layout.addWidget(self.auto_fill_blip)
        
        layout.addWidget(auto_group)
        
        # Conflict Resolution
        conflict_group = QGroupBox("Conflict Resolution")
        conflict_layout = QVBoxLayout(conflict_group)
        
        self.prefer_xmp = QCheckBox("Prefer XMP metadata over existing data")
        self.prefer_xmp.setToolTip("If enabled, XMP metadata will overwrite existing FileRecord data. If disabled, it only fills empty fields.")
        conflict_layout.addWidget(self.prefer_xmp)
        
        layout.addWidget(conflict_group)
        
        # Info
        info_label = QLabel(
            "Note: These settings control how metadata is processed during Phase 2 (Metadata) "
            "and Phase 3 (AI) of the extraction pipeline."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-style: italic; margin-top: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
