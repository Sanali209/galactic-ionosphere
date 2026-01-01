"""
Thumbnail Settings Page

Thumbnail generation, cache, and quality settings.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QSlider, QLineEdit, QFileDialog
)
from PySide6.QtCore import Qt


class ThumbnailSettingsPage(QWidget):
    """Thumbnail generation settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Size
        size_group = QGroupBox("Thumbnail Size")
        size_layout = QFormLayout(size_group)
        
        from PySide6.QtWidgets import QComboBox
        self.thumb_size_combo = QComboBox()
        self.thumb_size_combo.addItems(["Small (64px)", "Medium (128px)", "Large (256px)", "XLarge (512px)"])
        self.thumb_size_combo.setCurrentIndex(1)
        size_layout.addRow("Default Size:", self.thumb_size_combo)
        
        layout.addWidget(size_group)
        
        # Cache
        cache_group = QGroupBox("Cache")
        cache_layout = QFormLayout(cache_group)
        
        cache_path_layout = QHBoxLayout()
        self.cache_path = QLineEdit()
        self.cache_path.setPlaceholderText("Default: ~/.ucorefs/cache")
        cache_path_layout.addWidget(self.cache_path)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_cache)
        cache_path_layout.addWidget(self.browse_btn)
        cache_layout.addRow("Cache Location:", cache_path_layout)
        
        self.max_cache_spin = QSpinBox()
        self.max_cache_spin.setRange(100, 10000)
        self.max_cache_spin.setValue(1000)
        self.max_cache_spin.setSuffix(" MB")
        cache_layout.addRow("Max Cache Size:", self.max_cache_spin)
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        cache_layout.addRow("", self.clear_cache_btn)
        
        layout.addWidget(cache_group)
        
        # Quality
        quality_group = QGroupBox("Quality")
        quality_layout = QFormLayout(quality_group)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(50, 100)
        self.quality_slider.setValue(85)
        quality_layout.addRow("JPEG Quality:", self.quality_slider)
        
        layout.addWidget(quality_group)
        layout.addStretch()
    
    def _browse_cache(self):
        path = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if path:
            self.cache_path.setText(path)
