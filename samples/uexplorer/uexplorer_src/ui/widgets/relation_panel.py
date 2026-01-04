"""
Relation Panel Widget for UExplorer.

Shows duplicates, similar files, and other relations.
"""
import asyncio
from typing import Optional, Any
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                                QPushButton, QLabel, QHBoxLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from loguru import logger
from bson import ObjectId

from src.ucorefs.relations.models import Relation


class RelationPanel(QWidget):
    """Panel showing file relations (duplicates, similar, etc.)."""
    
    file_selected = Signal(str)  # Emits file_id when clicked
    
    def __init__(self, locator: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.locator = locator
        self._current_file_id = None
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with actions
        header = QHBoxLayout()
        header.addWidget(QLabel("Relations"))
        header.addStretch()
        
        self.find_btn = QPushButton("Find Duplicates")
        self.find_btn.clicked.connect(self._find_duplicates)
        header.addWidget(self.find_btn)
        
        layout.addLayout(header)
        
        # Relations tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Type", "File", "Score"])
        self.tree.setColumnWidth(0, 100)
        self.tree.setColumnWidth(1, 200)
        self.tree.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; }")
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)
        
        # Status
        self.status = QLabel("No file selected")
        self.status.setStyleSheet("color: gray;")
        layout.addWidget(self.status)
    
    def set_file(self, file_id: str):
        """Set current file and load its relations."""
        self._current_file_id = file_id
        asyncio.ensure_future(self._load_relations())
    
    async def _load_relations(self):
        """Load relations for current file."""
        self.tree.clear()
        
        if not self._current_file_id:
            self.status.setText("No file selected")
            return
        
        try:
            oid = ObjectId(self._current_file_id)
            
            # Find relations where file is source or target
            as_source = await Relation.find({"source_id": oid, "is_valid": True})
            as_target = await Relation.find({"target_id": oid, "is_valid": True})
            
            all_relations = as_source + as_target
            
            if not all_relations:
                self.status.setText("No relations found")
                return
            
            # Group by type
            by_type = {}
            for rel in all_relations:
                type_key = f"{rel.relation_type}/{rel.subtype}" if rel.subtype else rel.relation_type
                if type_key not in by_type:
                    by_type[type_key] = []
                by_type[type_key].append(rel)
            
            # Build tree
            for type_key, relations in by_type.items():
                type_item = QTreeWidgetItem(self.tree, [type_key, "", ""])
                type_item.setForeground(0, QColor("#888888"))
                
                for rel in relations:
                    # Show the OTHER file (not the current one)
                    other_id = str(rel.target_id) if str(rel.source_id) == self._current_file_id else str(rel.source_id)
                    score = rel.payload.get("score", "")
                    
                    item = QTreeWidgetItem(type_item, ["", other_id[:12] + "...", str(score)])
                    item.setData(1, Qt.UserRole, other_id)
            
            self.tree.expandAll()
            self.status.setText(f"{len(all_relations)} relations")
            
        except Exception as e:
            logger.error(f"Failed to load relations: {e}")
            self.status.setText(f"Error: {e}")
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Navigate to related file."""
        file_id = item.data(1, Qt.UserRole)
        if file_id:
            self.file_selected.emit(file_id)
    
    def _find_duplicates(self):
        """Trigger duplicate finding for entire library."""
        logger.info("Find duplicates requested")
        self.status.setText("Finding duplicates... (not implemented)")
        # TODO: Call RelationService to find duplicates


class RelationTreeWidget(QTreeWidget):
    """Simple tree showing relation categories in navigation panel."""
    
    category_selected = Signal(str)  # Emits category name
    
    def __init__(self, locator: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.locator = locator
        
        self.setHeaderLabel("Relations")
        self.setStyleSheet("QTreeWidget { background: #2d2d2d; color: #cccccc; border: none; }")
        self.itemClicked.connect(self._on_click)
        
        asyncio.ensure_future(self._load_categories())
    
    async def _load_categories(self):
        """Load relation categories with counts."""
        self.clear()
        
        try:
            # Count duplicates
            duplicates = await Relation.find({"relation_type": "duplicate", "is_valid": True})
            dup_item = QTreeWidgetItem(self, [f"Duplicates ({len(duplicates)})"])
            dup_item.setData(0, Qt.UserRole, "duplicate")
            
            # Count similars
            similars = await Relation.find({"relation_type": "similar", "is_valid": True})
            sim_item = QTreeWidgetItem(self, [f"Similar ({len(similars)})"])
            sim_item.setData(0, Qt.UserRole, "similar")
            
            # Count wrong/ignored
            wrong = await Relation.find({"is_valid": False})
            if wrong:
                wrong_item = QTreeWidgetItem(self, [f"Ignored ({len(wrong)})"])
                wrong_item.setData(0, Qt.UserRole, "wrong")
            
        except Exception as e:
            logger.error(f"Failed to load relation categories: {e}")
            QTreeWidgetItem(self, ["(Error loading)"])
    
    def _on_click(self, item: QTreeWidgetItem, column: int):
        """Handle category click."""
        category = item.data(0, Qt.UserRole)
        if category:
            self.category_selected.emit(category)
