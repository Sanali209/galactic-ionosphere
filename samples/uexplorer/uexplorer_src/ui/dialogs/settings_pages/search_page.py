"""
Search Settings Page

Search defaults and FAISS vector search settings.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QComboBox, QSlider
)
from PySide6.QtCore import Qt


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
