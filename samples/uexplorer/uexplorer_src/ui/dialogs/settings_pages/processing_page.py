"""
Processing Settings Page

Batch sizes and worker configuration.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox
)


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
