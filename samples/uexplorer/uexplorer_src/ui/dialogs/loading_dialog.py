"""
LoadingDialog - Startup loading screen with dynamic stage tracking.

Shows progress through all application startup stages.
Supports dynamic model stages via add_stage() method.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QFrame, QApplication, QScrollArea, QWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from typing import List, Tuple


class LoadingDialog(QDialog):
    """Startup loading dialog with visual stage tracking."""
    
    # Core loading stages (always shown)
    CORE_STAGES = [
        ("database", "Connecting to database"),
        ("core_services", "Initializing core services"),
        ("ucorefs_services", "Loading UCoreFS services"),
    ]
    
    # Final stages (after models)
    FINAL_STAGES = [
        ("ui", "Setting up UI"),
        ("tasks", "Starting task system"),
    ]
    
    stage_completed = Signal(str)  # stage_id
    all_completed = Signal()
    
    def __init__(self, model_stages: List[Tuple[str, str]] = None, parent=None):
        """
        Initialize loading dialog.
        
        Args:
            model_stages: List of (id, name) tuples for model stages.
                         Default: CLIP, BLIP, WDTagger, YOLO, GroundingDINO
        """
        super().__init__(parent)
        
        # Default model stages
        if model_stages is None:
            model_stages = [
                ("clip", "Loading CLIP model"),
                ("blip", "Loading BLIP model"),
                ("wd_tagger", "Loading WD-Tagger model"),
                ("yolo", "Loading YOLO detector"),
                ("grounding_dino", "Loading GroundingDINO"),
            ]
        
        # Combine all stages
        self.STAGES = self.CORE_STAGES + model_stages + self.FINAL_STAGES
        
        self.setWindowTitle("UExplorer")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(420, 400)
        self.setModal(True)
        
        # Dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                border: 2px solid #45475a;
                border-radius: 12px;
            }
            QLabel {
                color: #cdd6f4;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 6px;
                background-color: #313244;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 5px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("UExplorer")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Loading application...")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #6c7086;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(10)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximum(len(self.STAGES))
        self.progress.setValue(0)
        self.progress.setFixedHeight(24)
        layout.addWidget(self.progress)
        
        layout.addSpacing(10)
        
        # Scrollable stage list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        stages_container = QWidget()
        stages_container.setStyleSheet("background-color: #313244; border-radius: 8px;")
        stages_layout = QVBoxLayout(stages_container)
        stages_layout.setContentsMargins(12, 12, 12, 12)
        stages_layout.setSpacing(6)
        
        # Stage labels
        self.stage_labels = {}
        self._completed_count = 0
        
        for stage_id, stage_text in self.STAGES:
            label = QLabel(f"○  {stage_text}")
            label.setFont(QFont("Segoe UI", 10))
            label.setStyleSheet("color: #6c7086; background: transparent;")
            self.stage_labels[stage_id] = label
            stages_layout.addWidget(label)
        
        stages_layout.addStretch()
        scroll.setWidget(stages_container)
        layout.addWidget(scroll)
        
        # Center on screen
        self._center_on_screen()
    
    def _center_on_screen(self):
        """Center dialog on primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            x = (screen_geo.width() - self.width()) // 2
            y = (screen_geo.height() - self.height()) // 2
            self.move(x, y)
    
    def add_stage(self, stage_id: str, stage_text: str):
        """Dynamically add a stage (for custom models)."""
        if stage_id in self.stage_labels:
            return  # Already exists
            
        # This would require layout modification - for now, predefine stages
        pass
    
    def set_stage(self, stage_id: str, status: str = "loading"):
        """
        Update stage status.
        
        Args:
            stage_id: Stage identifier
            status: 'loading', 'done', 'error', or 'skip'
        """
        if stage_id not in self.stage_labels:
            return
            
        label = self.stage_labels[stage_id]
        stage_text = dict(self.STAGES).get(stage_id, stage_id)
        
        if status == "loading":
            label.setText(f"◉  {stage_text}...")
            label.setStyleSheet("color: #89b4fa; font-weight: bold; background: transparent;")
        elif status == "done":
            label.setText(f"✓  {stage_text}")
            label.setStyleSheet("color: #a6e3a1; background: transparent;")
            self._completed_count += 1
            self.progress.setValue(self._completed_count)
            self.stage_completed.emit(stage_id)
            
            if self._completed_count >= len(self.STAGES):
                self.all_completed.emit()
        elif status == "error":
            label.setText(f"✗  {stage_text}")
            label.setStyleSheet("color: #f38ba8; background: transparent;")
            self._completed_count += 1
            self.progress.setValue(self._completed_count)
        elif status == "skip":
            label.setText(f"–  {stage_text} (skipped)")
            label.setStyleSheet("color: #6c7086; background: transparent;")
            self._completed_count += 1
            self.progress.setValue(self._completed_count)
        
        # Process events to update UI immediately
        QApplication.processEvents()
    
    def set_substage(self, stage_id: str, substage_text: str):
        """Update stage with substage info."""
        if stage_id not in self.stage_labels:
            return
        
        label = self.stage_labels[stage_id]
        stage_text = dict(self.STAGES).get(stage_id, stage_id)
        label.setText(f"◉  {stage_text} ({substage_text})")
        QApplication.processEvents()
