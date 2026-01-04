"""
AI Settings Page

Embedding models, object detection, and hardware settings.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QSpinBox, QComboBox, QSlider
)
from PySide6.QtCore import Qt


class AISettingsPage(QWidget):
    """AI/Embedding and Detection settings."""
    
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
