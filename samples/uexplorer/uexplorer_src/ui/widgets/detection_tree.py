"""
Detection Tree Widget.
Displays a hierarchical tree of detection classes and groups.
Allows filtering by checking/unchecking items.
"""
from typing import List, Dict, Set, Optional
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHeaderView
from PySide6.QtCore import Signal, Qt

from loguru import logger

class DetectionTreeWidget(QTreeWidget):
    """
    Tree view for Detections.
    Hierarchy: Class -> Group.
    """
    
    # Emits list of active filters
    filter_changed = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Detection Class", "Count"])
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.setSortingEnabled(True)
        self.itemChanged.connect(self._on_item_changed)
        
        # Internal state to prevent signal loops
        self._updating = False
        
    def load_data(self, class_counts: List[Dict]):
        """
        Populate tree with aggregated data.
        Args:
            class_counts: List of dicts with keys: 'class_name', 'group_name', 'count', 'class_id'
        """
        self._updating = True
        self.clear()
        
        # Organize by class -> groups
        hierarchy = {}
        
        for item in class_counts:
            c_name = item.get('class_name', 'Unknown')
            g_name = item.get('group_name', 'any')
            count = item.get('count', 0)
            
            if c_name not in hierarchy:
                hierarchy[c_name] = {'total': 0, 'groups': {}}
            
            hierarchy[c_name]['total'] += count
            if g_name not in hierarchy[c_name]['groups']:
                hierarchy[c_name]['groups'][g_name] = 0
            hierarchy[c_name]['groups'][g_name] += count
            
        # Build Tree items
        for c_name, data in sorted(hierarchy.items()):
            class_item = QTreeWidgetItem(self)
            class_item.setText(0, c_name)
            class_item.setText(1, str(data['total']))
            class_item.setFlags(class_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            class_item.setCheckState(0, Qt.CheckState.Unchecked)
            
            # Groups (if any significant ones)
            groups = data['groups']
            if len(groups) > 1 or (len(groups) == 1 and 'any' not in groups):
                for g_name, g_count in sorted(groups.items()):
                    group_item = QTreeWidgetItem(class_item)
                    group_item.setText(0, g_name)
                    group_item.setText(1, str(g_count))
                    group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    group_item.setCheckState(0, Qt.CheckState.Unchecked)
            
        self._updating = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle checkbox toggles."""
        if self._updating:
            return
            
        if column == 0:
            self._updating = True
            
            # Propagate check state to children
            state = item.checkState(0)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
            
            # Emit filters
            self._emit_filters()
            
            self._updating = False

    def _emit_filters(self):
        """Collect checked items and emit filter list."""
        filters = []
        
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            class_item = root.child(i)
            
            # If class is checked
            if class_item.checkState(0) == Qt.CheckState.Checked:
                # Add generic filter for this class
                filters.append({
                    "class_name": class_item.text(0),
                    "group_name": "any", 
                    "min_count": 1,
                    "negate": False
                })
            elif class_item.checkState(0) == Qt.CheckState.PartiallyChecked:
                # Check children
                for j in range(class_item.childCount()):
                    group_item = class_item.child(j)
                    if group_item.checkState(0) == Qt.CheckState.Checked:
                        filters.append({
                            "class_name": class_item.text(0),
                            "group_name": group_item.text(0),
                            "min_count": 1,
                            "negate": False
                        })
                        
        logger.debug(f"[DetectionTree] Emitting {len(filters)} detection filters")
        self.filter_changed.emit(filters)

    def set_active_filters(self, filters: List[Dict]):
        """Sync check states with external filters."""
        self._updating = True
        
        # Reset all
        root = self.invisibleRootItem()
        
        # Helper to uncheck all
        def uncheck_recursive(item):
            item.setCheckState(0, Qt.CheckState.Unchecked)
            for i in range(item.childCount()):
                uncheck_recursive(item.child(i))
                
        for i in range(root.childCount()):
            uncheck_recursive(root.child(i))
            
        # Apply filters
        for f in filters:
            if f.get('negate'):
                continue # Tree only supports include for now
                
            c_name = f.get('class_name')
            g_name = f.get('group_name', 'any')
            
            # Find class item
            items = self.findItems(c_name, Qt.MatchFlag.MatchRecursive, 0)
            for item in items:
                # Is it a top level class item?
                if item.parent() is None:
                    if g_name == "any":
                        item.setCheckState(0, Qt.CheckState.Checked)
                        # Check children too
                        for j in range(item.childCount()):
                            item.child(j).setCheckState(0, Qt.CheckState.Checked)
                    else:
                        # Find group child
                        for j in range(item.childCount()):
                            child = item.child(j)
                            if child.text(0) == g_name:
                                child.setCheckState(0, Qt.CheckState.Checked)
                                item.setCheckState(0, Qt.CheckState.PartiallyChecked)
        
        self._updating = False
