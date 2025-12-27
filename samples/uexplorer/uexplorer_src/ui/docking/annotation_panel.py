"""
Dockable Annotation Panel for UExplorer.

Provides annotation workflow UI for ML training data curation.
Supports binary, multiclass, and multilabel annotation jobs.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QProgressBar, QListWidget, QListWidgetItem, QStackedWidget,
    QButtonGroup, QCheckBox, QWidget, QMessageBox, QInputDialog
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap
import asyncio
from pathlib import Path
from loguru import logger

from uexplorer_src.ui.docking.panel_base import PanelBase


class AnnotationPanel(PanelBase):
    """
    Dockable panel for annotation workflow.
    
    Features:
    - Job CRUD (Create, Read, Update, Delete)
    - File navigation (Prev/Next/Skip)
    - Binary/Multiclass/Multilabel annotation
    - Progress tracking
    - Export to JSON/CSV
    """
    
    # Emitted when file is annotated
    file_annotated = Signal(str, str)  # file_id, value
    
    def __init__(self, parent, locator):
        self._annotation_service = None
        self._thumbnail_service = None
        self._current_job = None
        self._current_record = None
        self._current_file = None
        super().__init__(locator, parent)
        
        # Get services
        try:
            from src.ucorefs.annotation.service import AnnotationService
            self._annotation_service = locator.get_system(AnnotationService)
        except (KeyError, ImportError):
            logger.warning("AnnotationService not available")
        
        try:
            from src.ucorefs.thumbnails.service import ThumbnailService
            self._thumbnail_service = locator.get_system(ThumbnailService)
        except (KeyError, ImportError):
            pass
    
    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        
        # Header with job selector
        header = QHBoxLayout()
        
        title = QLabel("📋 Annotation")
        title.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 14px;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Job management buttons
        self.btn_new_job = QPushButton("+")
        self.btn_new_job.setFixedSize(24, 24)
        self.btn_new_job.setToolTip("New Job")
        self.btn_new_job.clicked.connect(self._on_new_job)
        header.addWidget(self.btn_new_job)
        
        self.btn_edit_job = QPushButton("✎")
        self.btn_edit_job.setFixedSize(24, 24)
        self.btn_edit_job.setToolTip("Edit Job")
        self.btn_edit_job.clicked.connect(self._on_edit_job)
        header.addWidget(self.btn_edit_job)
        
        self.btn_delete_job = QPushButton("🗑")
        self.btn_delete_job.setFixedSize(24, 24)
        self.btn_delete_job.setToolTip("Delete Job")
        self.btn_delete_job.clicked.connect(self._on_delete_job)
        header.addWidget(self.btn_delete_job)
        
        layout.addLayout(header)
        
        # Job selector dropdown
        job_layout = QHBoxLayout()
        job_layout.addWidget(QLabel("Job:"))
        
        self.job_combo = QComboBox()
        self.job_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self.job_combo.currentIndexChanged.connect(self._on_job_selected)
        job_layout.addWidget(self.job_combo, 1)
        
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.setToolTip("Add selected files to job")
        self.btn_add_files.clicked.connect(self._on_add_files)
        job_layout.addWidget(self.btn_add_files)
        
        layout.addLayout(job_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m (%p%)")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0d6efd;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Current file display
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setMinimumHeight(150)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        layout.addWidget(self.thumbnail_label)
        
        # Choice area (stacked widget for different job types)
        self.choice_stack = QStackedWidget()
        
        # Binary choices
        self.binary_widget = QWidget()
        binary_layout = QHBoxLayout(self.binary_widget)
        self.btn_choice_1 = QPushButton("Yes")
        self.btn_choice_1.setStyleSheet("background-color: #27ae60; color: white; padding: 10px;")
        self.btn_choice_1.clicked.connect(lambda: self._annotate(0))
        self.btn_choice_2 = QPushButton("No")
        self.btn_choice_2.setStyleSheet("background-color: #c0392b; color: white; padding: 10px;")
        self.btn_choice_2.clicked.connect(lambda: self._annotate(1))
        binary_layout.addWidget(self.btn_choice_1)
        binary_layout.addWidget(self.btn_choice_2)
        self.choice_stack.addWidget(self.binary_widget)
        
        # Multiclass choices (button group)
        self.multiclass_widget = QWidget()
        self.multiclass_layout = QVBoxLayout(self.multiclass_widget)
        self.multiclass_group = QButtonGroup()
        self.choice_stack.addWidget(self.multiclass_widget)
        
        # Multilabel choices (checkboxes)
        self.multilabel_widget = QWidget()
        self.multilabel_layout = QVBoxLayout(self.multilabel_widget)
        self.multilabel_checkboxes = []
        self.choice_stack.addWidget(self.multilabel_widget)
        
        layout.addWidget(self.choice_stack)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("← Prev")
        self.btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setStyleSheet("background-color: #555555;")
        self.btn_skip.clicked.connect(self._on_skip)
        nav_layout.addWidget(self.btn_skip)
        
        self.btn_next = QPushButton("Next →")
        self.btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        # Export button
        self.btn_export = QPushButton("📤 Export Annotations")
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)
        
        # Status
        self.status_label = QLabel("Select or create an annotation job")
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Initial load
        asyncio.ensure_future(self._load_jobs())
    
    # ==================== Job CRUD ====================
    
    async def _load_jobs(self):
        """Load available annotation jobs."""
        if not self._annotation_service:
            return
        
        try:
            jobs = await self._annotation_service.list_jobs()
            self.job_combo.clear()
            self.job_combo.addItem("-- Select Job --", None)
            
            for job in jobs:
                self.job_combo.addItem(
                    f"{job.name} ({job.job_type})",
                    str(job._id)
                )
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
    
    def _on_new_job(self):
        """Create new annotation job."""
        name, ok = QInputDialog.getText(self, "New Job", "Job name:")
        if not ok or not name:
            return
        
        # Ask for type
        job_type, ok = QInputDialog.getItem(
            self, "Job Type", "Select type:",
            ["binary", "multiclass", "multilabel"],
            0, False
        )
        if not ok:
            return
        
        # Ask for choices
        choices_str, ok = QInputDialog.getText(
            self, "Choices", 
            "Enter choices (comma-separated):",
            text="yes,no" if job_type == "binary" else ""
        )
        if not ok:
            return
        
        choices = [c.strip() for c in choices_str.split(",") if c.strip()]
        
        async def create():
            try:
                await self._annotation_service.create_job(
                    name=name,
                    job_type=job_type,
                    choices=choices
                )
                await self._load_jobs()
                self.status_label.setText(f"Created job: {name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create job: {e}")
        
        asyncio.ensure_future(create())
    
    def _on_edit_job(self):
        """Edit selected job."""
        job_id = self.job_combo.currentData()
        if not job_id:
            return
        
        QMessageBox.information(
            self, "Edit Job",
            "Edit functionality coming soon.\nUse delete and recreate for now."
        )
    
    def _on_delete_job(self):
        """Delete selected job."""
        job_id = self.job_combo.currentData()
        if not job_id:
            return
        
        reply = QMessageBox.question(
            self, "Delete Job",
            "Delete this job and all annotations?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        async def delete():
            try:
                from bson import ObjectId
                await self._annotation_service.delete_job(ObjectId(job_id))
                await self._load_jobs()
                self.status_label.setText("Job deleted")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
        
        asyncio.ensure_future(delete())
    
    def _on_job_selected(self, index):
        """Handle job selection."""
        job_id = self.job_combo.currentData()
        if not job_id:
            self._current_job = None
            return
        
        asyncio.ensure_future(self._load_job(job_id))
    
    async def _load_job(self, job_id: str):
        """Load job details and first file."""
        try:
            from bson import ObjectId
            
            job = await self._annotation_service.get_job(ObjectId(job_id))
            if not job:
                return
            
            self._current_job = job
            
            # Update progress
            progress = await self._annotation_service.get_job_progress(ObjectId(job_id))
            self.progress_bar.setMaximum(progress.get("total", 0))
            self.progress_bar.setValue(progress.get("annotated", 0))
            
            # Setup choices UI based on job type
            self._setup_choices_ui(job)
            
            # Load first unannotated file
            await self._load_next_file()
            
        except Exception as e:
            logger.error(f"Failed to load job: {e}")
    
    def _setup_choices_ui(self, job):
        """Setup choice buttons based on job type."""
        if job.job_type == "binary":
            self.choice_stack.setCurrentIndex(0)
            if len(job.choices) >= 2:
                self.btn_choice_1.setText(job.choices[0])
                self.btn_choice_2.setText(job.choices[1])
        elif job.job_type == "multiclass":
            self.choice_stack.setCurrentIndex(1)
            # Clear and rebuild buttons
            for btn in self.multiclass_group.buttons():
                self.multiclass_layout.removeWidget(btn)
                btn.deleteLater()
            
            for i, choice in enumerate(job.choices):
                btn = QPushButton(choice)
                btn.clicked.connect(lambda checked, idx=i: self._annotate(idx))
                self.multiclass_layout.addWidget(btn)
                self.multiclass_group.addButton(btn, i)
        else:
            self.choice_stack.setCurrentIndex(2)
            # Clear and rebuild checkboxes
            for cb in self.multilabel_checkboxes:
                self.multilabel_layout.removeWidget(cb)
                cb.deleteLater()
            self.multilabel_checkboxes.clear()
            
            for choice in job.choices:
                cb = QCheckBox(choice)
                self.multilabel_layout.addWidget(cb)
                self.multilabel_checkboxes.append(cb)
            
            # Add submit button for multilabel
            submit_btn = QPushButton("Submit")
            submit_btn.clicked.connect(self._submit_multilabel)
            self.multilabel_layout.addWidget(submit_btn)
    
    # ==================== File Management ====================
    
    def _on_add_files(self):
        """Add selected files to current job."""
        if not self._current_job:
            QMessageBox.information(self, "Add Files", "Select a job first.")
            return
        
        QMessageBox.information(
            self, "Add Files",
            "To add files: Select files in the file browser, then use this button.\n\n"
            "Coming soon: Integration with SelectionManager"
        )
    
    async def _load_next_file(self):
        """Load next unannotated file."""
        if not self._current_job or not self._annotation_service:
            return
        
        try:
            result = await self._annotation_service.get_next_unannotated(
                self._current_job._id
            )
            
            if not result:
                self.file_label.setText("All files annotated!")
                self.thumbnail_label.clear()
                self._current_record = None
                self._current_file = None
                return
            
            self._current_record = result.get("record")
            self._current_file = result.get("file")
            
            if self._current_file:
                self.file_label.setText(self._current_file.name)
                await self._load_thumbnail(self._current_file)
            
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
    
    async def _load_thumbnail(self, file_record):
        """Load thumbnail for current file."""
        if not self._thumbnail_service:
            return
        
        try:
            thumb_path = await self._thumbnail_service.get_or_create(
                file_record._id, size=256
            )
            if thumb_path and Path(thumb_path).exists():
                pixmap = QPixmap(str(thumb_path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        200, 150,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.thumbnail_label.setPixmap(scaled)
        except Exception as e:
            logger.debug(f"Thumbnail failed: {e}")
    
    # ==================== Annotation ====================
    
    def _annotate(self, choice_index: int):
        """Annotate current file with choice."""
        if not self._current_job or not self._current_file:
            return
        
        value = self._current_job.choices[choice_index] if choice_index < len(self._current_job.choices) else str(choice_index)
        asyncio.ensure_future(self._save_annotation(value))
    
    def _submit_multilabel(self):
        """Submit multilabel annotation."""
        if not self._current_job:
            return
        
        selected = [
            cb.text() for cb in self.multilabel_checkboxes
            if cb.isChecked()
        ]
        asyncio.ensure_future(self._save_annotation(selected))
    
    async def _save_annotation(self, value):
        """Save annotation and move to next."""
        try:
            from bson import ObjectId
            
            await self._annotation_service.annotate(
                self._current_job._id,
                self._current_file._id,
                value
            )
            
            self.file_annotated.emit(str(self._current_file._id), str(value))
            
            # Update progress
            progress = await self._annotation_service.get_job_progress(
                self._current_job._id
            )
            self.progress_bar.setValue(progress.get("annotated", 0))
            
            # Load next
            await self._load_next_file()
            
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def _on_skip(self):
        """Skip current file."""
        if not self._current_job or not self._current_file:
            return
        
        async def skip():
            await self._annotation_service.skip(
                self._current_job._id,
                self._current_file._id
            )
            await self._load_next_file()
        
        asyncio.ensure_future(skip())
    
    def _on_prev(self):
        """Go to previous file."""
        QMessageBox.information(self, "Previous", "Coming soon")
    
    def _on_next(self):
        """Go to next file."""
        asyncio.ensure_future(self._load_next_file())
    
    # ==================== Export ====================
    
    def _on_export(self):
        """Export annotations."""
        if not self._current_job:
            QMessageBox.information(self, "Export", "Select a job first.")
            return
        
        from PySide6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations",
            f"{self._current_job.name}_annotations.json",
            "JSON (*.json);;CSV (*.csv)"
        )
        
        if not file_path:
            return
        
        async def export():
            try:
                data = await self._annotation_service.export_annotations(
                    self._current_job._id
                )
                
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(
                    self, "Exported",
                    f"Exported {len(data)} annotations to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
        
        asyncio.ensure_future(export())
    
    def on_update(self, context=None):
        """Called when panel is updated."""
        pass
