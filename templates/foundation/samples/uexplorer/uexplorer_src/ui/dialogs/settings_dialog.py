"""
UExplorer - Settings Dialog

Centralized settings dialog for application configuration.
Organized by sections: General, Thumbnails, AI, Search, Processing.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QStackedWidget, QWidget,
    QCheckBox, QSpinBox, QComboBox, QLineEdit, QGroupBox,
    QFormLayout, QSlider, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from loguru import logger


class GeneralSettingsPage(QWidget):
    """General application settings."""
    
    def __init__(self, parent=None):
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


class ThumbnailSettingsPage(QWidget):
    """Thumbnail generation settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Size
        size_group = QGroupBox("Thumbnail Size")
        size_layout = QFormLayout(size_group)
        
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


class AISettingsPage(QWidget):
    """AI/Embedding settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Models
        models_group = QGroupBox("AI Models")
        models_layout = QVBoxLayout(models_group)
        
        self.enable_clip = QCheckBox("Enable CLIP (image embeddings)")
        self.enable_clip.setChecked(True)
        models_layout.addWidget(self.enable_clip)
        
        self.enable_blip = QCheckBox("Enable BLIP (image captioning)")
        self.enable_blip.setChecked(True)
        models_layout.addWidget(self.enable_blip)
        
        self.enable_dino = QCheckBox("Enable GroundingDINO (object detection)")
        self.enable_dino.setChecked(True)
        models_layout.addWidget(self.enable_dino)
        
        layout.addWidget(models_group)
        
        # Hardware
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QFormLayout(hardware_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (GPU if available)", "CPU Only", "CUDA", "DirectML"])
        hardware_layout.addRow("Compute Device:", self.device_combo)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        hardware_layout.addRow("Batch Size:", self.batch_size_spin)
        
        layout.addWidget(hardware_group)
        layout.addStretch()


class SearchSettingsPage(QWidget):
    """Search/FAISS settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Defaults
        defaults_group = QGroupBox("Default Search")
        defaults_layout = QFormLayout(defaults_group)
        
        self.default_limit = QSpinBox()
        self.default_limit.setRange(10, 1000)
        self.default_limit.setValue(100)
        defaults_layout.addRow("Default Result Limit:", self.default_limit)
        
        self.similarity_threshold = QSlider(Qt.Horizontal)
        self.similarity_threshold.setRange(50, 100)
        self.similarity_threshold.setValue(70)
        defaults_layout.addRow("Min Similarity (%):", self.similarity_threshold)
        
        layout.addWidget(defaults_group)
        
        # FAISS
        faiss_group = QGroupBox("Vector Search (FAISS)")
        faiss_layout = QFormLayout(faiss_group)
        
        self.index_type = QComboBox()
        self.index_type.addItems(["Flat (Exact)", "IVF (Approximate)", "HNSW (Fast)"])
        faiss_layout.addRow("Index Type:", self.index_type)
        
        self.rebuild_btn = QPushButton("Rebuild Index")
        faiss_layout.addRow("", self.rebuild_btn)
        
        layout.addWidget(faiss_group)
        layout.addStretch()


class ProcessingSettingsPage(QWidget):
    """Processing pipeline settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Batch sizes
        batch_group = QGroupBox("Batch Sizes")
        batch_layout = QFormLayout(batch_group)
        
        self.phase1_batch = QSpinBox()
        self.phase1_batch.setRange(50, 500)
        self.phase1_batch.setValue(200)
        batch_layout.addRow("Phase 1 (Discovery):", self.phase1_batch)
        
        self.phase2_batch = QSpinBox()
        self.phase2_batch.setRange(5, 100)
        self.phase2_batch.setValue(20)
        batch_layout.addRow("Phase 2 (Enrichment):", self.phase2_batch)
        
        self.phase3_batch = QSpinBox()
        self.phase3_batch.setRange(1, 10)
        self.phase3_batch.setValue(1)
        batch_layout.addRow("Phase 3 (AI):", self.phase3_batch)
        
        layout.addWidget(batch_group)
        
        # Workers
        workers_group = QGroupBox("Background Workers")
        workers_layout = QFormLayout(workers_group)
        
        self.worker_count = QSpinBox()
        self.worker_count.setRange(1, 16)
        self.worker_count.setValue(3)
        workers_layout.addRow("Worker Threads:", self.worker_count)
        
        layout.addWidget(workers_group)
        layout.addStretch()


class SettingsDialog(QDialog):
    """
    Centralized settings dialog.
    
    Sections:
    - General: Theme, startup options
    - Thumbnails: Size, cache, quality
    - AI/Embeddings: Model enable/disable, hardware
    - Search: Defaults, FAISS settings
    - Processing: Batch sizes, workers
    """
    
    settings_changed = Signal()
    
    def __init__(self, locator=None, parent=None):
        super().__init__(parent)
        self._locator = locator
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(700, 500)
        self.setup_ui()
        
        logger.info("SettingsDialog opened")
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QHBoxLayout(self)
        
        # Left: Category list
        self.category_list = QListWidget()
        self.category_list.setFixedWidth(150)
        self.category_list.currentRowChanged.connect(self._on_category_changed)
        
        categories = [
            ("General", "⚙️"),
            ("Thumbnails", "🖼️"),
            ("AI / Embeddings", "🤖"),
            ("Search", "🔍"),
            ("Processing", "⚡"),
        ]
        
        for name, icon in categories:
            item = QListWidgetItem(f"{icon} {name}")
            self.category_list.addItem(item)
        
        layout.addWidget(self.category_list)
        
        # Right: Stacked pages
        right_layout = QVBoxLayout()
        
        self.stack = QStackedWidget()
        self.stack.addWidget(GeneralSettingsPage())
        self.stack.addWidget(ThumbnailSettingsPage())
        self.stack.addWidget(AISettingsPage())
        self.stack.addWidget(SearchSettingsPage())
        self.stack.addWidget(ProcessingSettingsPage())
        
        right_layout.addWidget(self.stack, 1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #5a7aaa;
                color: white;
            }
        """)
        btn_layout.addWidget(self.btn_ok)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        right_layout.addLayout(btn_layout)
        layout.addLayout(right_layout, 1)
        
        # Select first category
        self.category_list.setCurrentRow(0)
        
        # Apply theme
        self._apply_style()
    
    def _apply_style(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QListWidget {
                background-color: #3d3d3d;
                color: #ffffff;
                border: none;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 12px 8px;
            }
            QListWidget::item:selected {
                background-color: #5a7aaa;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QCheckBox {
                color: #cccccc;
            }
            QComboBox, QSpinBox, QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #cccccc;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: #5a7aaa;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
    
    def _on_category_changed(self, row):
        """Handle category selection."""
        self.stack.setCurrentIndex(row)
    
    def _on_apply(self):
        """Apply settings without closing."""
        self._save_settings()
        self.settings_changed.emit()
        logger.info("Settings applied")
    
    def _on_ok(self):
        """Apply and close."""
        self._save_settings()
        self.settings_changed.emit()
        self.accept()
    
    def _save_settings(self):
        """Save settings to config."""
        # TODO: Save to actual config
        logger.info("Settings saved")
