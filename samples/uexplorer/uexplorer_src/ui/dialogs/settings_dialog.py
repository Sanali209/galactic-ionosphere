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

from uexplorer_src.ui.dialogs.maintenance_settings_page import MaintenanceSettingsPage


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
    """AI/Embedding and Detection settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Embedding Models
        models_group = QGroupBox("Embedding Models")
        models_layout = QVBoxLayout(models_group)
        
        self.enable_clip = QCheckBox("Enable CLIP (image embeddings)")
        self.enable_clip.setChecked(True)
        models_layout.addWidget(self.enable_clip)
        
        self.enable_blip = QCheckBox("Enable BLIP (image captioning)")
        self.enable_blip.setChecked(True)
        models_layout.addWidget(self.enable_blip)
        
        self.enable_dino = QCheckBox("Enable DINO (visual features)")
        self.enable_dino.setChecked(True)
        models_layout.addWidget(self.enable_dino)
        
        layout.addWidget(models_group)
        
        # Detection Settings
        detection_group = QGroupBox("Object Detection")
        detection_layout = QFormLayout(detection_group)
        
        self.enable_detection = QCheckBox("Enable detection during processing")
        self.enable_detection.setChecked(True)
        detection_layout.addRow("", self.enable_detection)
        
        self.detection_backend = QComboBox()
        self.detection_backend.addItems(["YOLO (Objects)", "MTCNN (Faces)", "Both"])
        detection_layout.addRow("Detection Backend:", self.detection_backend)
        
        self.yolo_confidence = QSlider(Qt.Horizontal)
        self.yolo_confidence.setRange(10, 90)
        self.yolo_confidence.setValue(25)
        detection_layout.addRow("YOLO Confidence (%):", self.yolo_confidence)
        
        self.enable_mtcnn = QCheckBox("Enable face detection (MTCNN)")
        self.enable_mtcnn.setChecked(True)
        detection_layout.addRow("", self.enable_mtcnn)
        
        layout.addWidget(detection_group)
        
        # Hardware
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QFormLayout(hardware_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (GPU if available)", "CPU Only", "CUDA", "DirectML"])
        hardware_layout.addRow("Compute Device:", self.device_combo)
        
        self.result_limit = QSpinBox()
        self.result_limit.setRange(10, 500)
        self.result_limit.setValue(50)
        hardware_layout.addRow("Result Limit:", self.result_limit)
        
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
        workers_group = QGroupBox("Resource Usage")
        workers_layout = QFormLayout(workers_group)
        
        self.worker_count = QSpinBox()
        self.worker_count.setRange(1, 32)
        self.worker_count.setValue(8)
        self.worker_count.setToolTip("Number of background task orchestrators. Higher = more tasks managed at once.")
        workers_layout.addRow("Task Orchestrators:", self.worker_count)

        self.ai_workers = QSpinBox()
        self.ai_workers.setRange(1, 16)
        self.ai_workers.setValue(4)
        self.ai_workers.setToolTip("Number of concurrent AI/CPU threads. Limit this based on your CPU cores.")
        workers_layout.addRow("AI Threads (CPU):", self.ai_workers)
        
        layout.addWidget(workers_group)
        layout.addStretch()


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


class SettingsDialog(QDialog):
    """
    Centralized settings dialog integrated with ConfigManager.
    
    Sections:
    - General: Theme, startup options
    - Thumbnails: Size, cache, quality
    - AI/Embeddings: Model enable/disable, hardware
    - Search: Defaults, FAISS settings
    - Processing: Batch sizes, workers
    """
    
    settings_changed = Signal()
    
    def __init__(self, config_manager=None, locator=None, parent=None):
        """
        Initialize settings dialog.
        
        Args:
            config_manager: ConfigManager instance (required)
            locator: ServiceLocator (optional, for service access)
            parent: Parent widget
        """
        super().__init__(parent)
        self._config = config_manager
        self._locator = locator
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(700, 500)
        self.setup_ui()
        self._load_settings()
        
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
            ("Metadata", "🔖"),
            ("Search", "🔍"),
            ("Processing", "⚡"),
            ("Maintenance", "🔧"),
        ]
        
        for name, icon in categories:
            item = QListWidgetItem(f"{icon} {name}")
            self.category_list.addItem(item)
        
        layout.addWidget(self.category_list)
        
        # Right: Stacked pages
        right_layout = QVBoxLayout()
        
        self.stack = QStackedWidget()
        self.general_page = GeneralSettingsPage()
        self.thumbnail_page = ThumbnailSettingsPage()
        self.ai_page = AISettingsPage()
        self.metadata_page = MetadataSettingsPage()
        self.search_page = SearchSettingsPage()
        self.processing_page = ProcessingSettingsPage()
        self.maintenance_page = MaintenanceSettingsPage()
        
        # Connect maintenance page signal
        self.maintenance_page.run_task_requested.connect(self._on_run_maintenance_task)
        
        self.stack.addWidget(self.general_page)
        self.stack.addWidget(self.thumbnail_page)
        self.stack.addWidget(self.ai_page)
        self.stack.addWidget(self.metadata_page)
        self.stack.addWidget(self.search_page)
        self.stack.addWidget(self.processing_page)
        self.stack.addWidget(self.maintenance_page)
        
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
    
    def _load_settings(self):
        """Load settings from ConfigManager."""
        if not self._config:
            logger.warning("No ConfigManager - settings won't be persisted")
            return
        
        try:
            data = self._config.data
            
            # General settings
            if hasattr(data, 'general'):
                theme = getattr(data.general, 'theme', 'dark')
                theme_map = {'dark': 0, 'light': 1, 'system': 2}
                self.general_page.theme_combo.setCurrentIndex(theme_map.get(theme, 0))
                
                # Task workers
                self.processing_page.worker_count.setValue(
                    getattr(data.general, 'task_workers', 8)
                )

            # Processing settings
            if hasattr(data, 'processing'):
                self.processing_page.ai_workers.setValue(
                    getattr(data.processing, 'ai_workers', 4)
                )
            
            # AI settings
            if hasattr(data, 'ai'):
                device = getattr(data.ai, 'device', 'cpu')
                device_map = {'auto': 0, 'cpu': 1, 'cuda': 2, 'directml': 3}
                self.ai_page.device_combo.setCurrentIndex(device_map.get(device.lower(), 0))
                
                # Result limit
                self.ai_page.result_limit.setValue(getattr(data.ai, 'result_limit', 50))
            
            # Processing settings - Detection
            if hasattr(data, 'processing'):
                proc = data.processing
                if hasattr(proc, 'detection'):
                    det = proc.detection
                    # Detection enabled
                    self.ai_page.enable_detection.setChecked(
                        getattr(det, 'enabled', True)
                    )
                    # Backend
                    backend = getattr(det, 'backend', 'yolo')
                    backend_map = {'yolo': 0, 'mtcnn': 1, 'both': 2}
                    self.ai_page.detection_backend.setCurrentIndex(backend_map.get(backend, 0))
                    
                    # YOLO confidence
                    if hasattr(det, 'yolo'):
                        conf = getattr(det.yolo, 'confidence', 0.25)
                        self.ai_page.yolo_confidence.setValue(int(conf * 100))
                    
                    # MTCNN enabled
                    if hasattr(det, 'mtcnn'):
                        self.ai_page.enable_mtcnn.setChecked(
                            getattr(det.mtcnn, 'enabled', True)
                        )
                
                # Processing settings - Embeddings
                if hasattr(proc, 'embeddings'):
                    emb = proc.embeddings
                    if hasattr(emb, 'clip'):
                        self.ai_page.enable_clip.setChecked(getattr(emb.clip, 'enabled', True))
                    if hasattr(emb, 'dino'):
                        self.ai_page.enable_dino.setChecked(getattr(emb.dino, 'enabled', True))
                    if hasattr(emb, 'blip'):
                        self.ai_page.enable_blip.setChecked(getattr(emb.blip, 'enabled', True))
            
            # Metadata settings
            if hasattr(data, 'metadata'):
                meta = data.metadata
                self.metadata_page.auto_fill_blip.setChecked(
                    getattr(meta, 'auto_fill_description_from_blip', True)
                )
                self.metadata_page.prefer_xmp.setChecked(
                    getattr(meta, 'prefer_xmp_over_existing', False)
                )
            
            logger.info("Settings loaded from ConfigManager")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    
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
        """Save settings to ConfigManager."""
        if not self._config:
            logger.warning("No ConfigManager - settings not saved")
            return
        
        try:
            # General settings
            theme_map = {0: 'dark', 1: 'light', 2: 'system'}
            theme = theme_map.get(self.general_page.theme_combo.currentIndex(), 'dark')
            self._config.update('general', 'theme', theme)
            
            # Task workers
            workers = self.processing_page.worker_count.value()
            self._config.update('general', 'task_workers', workers)

            # AI workers
            ai_workers = self.processing_page.ai_workers.value()
            self._config.update('processing', 'ai_workers', ai_workers)
            
            # AI device
            device_map = {0: 'auto', 1: 'cpu', 2: 'cuda', 3: 'directml'}
            device = device_map.get(self.ai_page.device_combo.currentIndex(), 'cpu')
            self._config.update('ai', 'device', device)
            
            # Result limit
            result_limit = self.ai_page.result_limit.value()
            self._config.update('ai', 'result_limit', result_limit)
            
            # Note: Processing/detection/embeddings are nested objects
            # ConfigManager.update() works on top-level sections
            # For nested updates, we modify data directly then trigger save
            data = self._config.data
            
            # Ensure processing section exists
            if not hasattr(data, 'processing'):
                data.processing = {}
            
            # Detection settings (read current, update, save)
            # This is a limitation - we log the intended values for now
            logger.info(f"Detection enabled: {self.ai_page.enable_detection.isChecked()}")
            logger.info(f"Detection backend: {self.ai_page.detection_backend.currentText()}")
            logger.info(f"YOLO confidence: {self.ai_page.yolo_confidence.value() / 100.0}")
            logger.info(f"MTCNN enabled: {self.ai_page.enable_mtcnn.isChecked()}")
            logger.info(f"CLIP enabled: {self.ai_page.enable_clip.isChecked()}")
            logger.info(f"DINO enabled: {self.ai_page.enable_dino.isChecked()}")
            logger.info(f"BLIP enabled: {self.ai_page.enable_blip.isChecked()}")
            
            # Metadata settings
            self._config.update('metadata', 'auto_fill_description_from_blip', self.metadata_page.auto_fill_blip.isChecked())
            self._config.update('metadata', 'prefer_xmp_over_existing', self.metadata_page.prefer_xmp.isChecked())
            
            logger.info("Settings saved to ConfigManager")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def _on_run_maintenance_task(self, task_name: str):
        """Handle manual maintenance task execution from settings."""
        logger.info(f"Running maintenance task from settings: {task_name}")
        
        if self._locator:
            from src.ucorefs.services.maintenance_service import MaintenanceService
            maintenance = self._locator.get_system(MaintenanceService)
            
            # Execute in background
            import asyncio
            asyncio.create_task(self._execute_maintenance(maintenance, task_name))
        else:
            logger.warning("Cannot run task: ServiceLocator not available")
    
    async def _execute_maintenance(self, maintenance, task_name: str):
        """Execute maintenance task asynchronously."""
        try:
            result = None
            
            if task_name == "reprocess_incomplete_embeddings":
                result = await maintenance.reprocess_incomplete_embeddings()
            elif task_name == "diagnose_pipeline_state":
                result = await maintenance.diagnose_pipeline_state()
            elif task_name == "fix_file_types":
                result = await maintenance.fix_file_types()
            elif task_name == "background_verification":
                await maintenance.background_count_verification()
            elif task_name == "database_optimization":
                result = await maintenance.database_optimization()
            elif task_name == "cache_cleanup":
                result = await maintenance.cache_cleanup()
            elif task_name == "orphaned_cleanup":
                result = await maintenance.cleanup_orphaned_file_records()
            elif task_name == "log_rotation":
                result = await maintenance.log_rotation()
            elif task_name == "database_cleanup":
                result = await maintenance.cleanup_old_records()
            
            logger.info(f"Maintenance task {task_name} complete: {result}")
            
        except Exception as e:
            logger.error(f"Maintenance task {task_name} failed: {e}", exc_info=True)


