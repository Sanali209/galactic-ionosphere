"""
UExplorer - Settings Dialog

Centralized settings dialog for application configuration.
Organized by sections: General, Thumbnails, AI, Search, Processing.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QStackedWidget, QWidget
)
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator
    from src.ucorefs.services.maintenance_service import MaintenanceService
from PySide6.QtCore import Signal
from loguru import logger

# Import page classes from settings_pages package
from uexplorer_src.ui.dialogs.settings_pages import (
    GeneralSettingsPage,
    ThumbnailSettingsPage,
    AISettingsPage,
    SearchSettingsPage,
    ProcessingSettingsPage,
    MetadataSettingsPage,
)
from uexplorer_src.ui.dialogs.maintenance_settings_page import MaintenanceSettingsPage


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
    
    
    def __init__(self, config_manager: Optional[Any] = None, locator: Optional["ServiceLocator"] = None, parent: Optional[QWidget] = None) -> None:
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
    
    def _on_category_changed(self, row: int) -> None:
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
        """Handle manual maintenance task execution from settings via EngineProxy."""
        logger.info(f"Running maintenance task from settings: {task_name}")
        
        if self._locator:
            from src.core.engine.proxy import EngineProxy
            engine_proxy = self._locator.get_system(EngineProxy)
            
            if not engine_proxy:
                logger.warning("EngineProxy not available")
                return
            
            # Execute on Engine thread via proxy
            import asyncio
            asyncio.create_task(self._execute_maintenance_via_proxy(engine_proxy, task_name))
        else:
            logger.warning("Cannot run task: ServiceLocator not available")
    
    async def _execute_maintenance_via_proxy(self, engine_proxy, task_name: str) -> None:
        """Execute maintenance task via EngineProxy."""
        try:
            async def _run_maintenance():
                from src.core.locator import get_active_locator
                from src.ucorefs.services.maintenance_service import MaintenanceService
                sl = get_active_locator()
                maintenance = sl.get_system(MaintenanceService)
                
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
                
                return result
            
            future = engine_proxy.submit(_run_maintenance())
            import asyncio
            result = await asyncio.wrap_future(future)
            
            logger.info(f"Maintenance task {task_name} complete: {result}")
            
        except Exception as e:

