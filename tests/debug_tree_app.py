
import sys
import os
import random

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtCore import QAbstractListModel, Qt, Slot, QModelIndex, QObject, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlContext

from src.ui.models.flat_tree import BaseFlatTreeModel

class MockTreeModel(BaseFlatTreeModel):
    def __init__(self):
        super().__init__()
        self._items = []
        # Init with some data
        self.reset_data()

    def roleNames(self):
        roles = super().roleNames()
        # Add custom roles if needed, but Base should suffice for "display" and "name"
        return roles
    
    def data(self, index, role=Qt.DisplayRole):
        return super().data(index, role)

    @Slot()
    def reset_data(self):
        self.beginResetModel()
        self._items = [
            {"name": "Root 1", "depth": 0, "expanded": False, "has_children": True, "id": "r1"},
            {"name": "Root 2", "depth": 0, "expanded": True, "has_children": True, "id": "r2"},
            {"name": "Child 2.1", "depth": 1, "expanded": False, "has_children": False, "id": "c21"},
            {"name": "Child 2.2", "depth": 1, "expanded": False, "has_children": False, "id": "c22"},
            {"name": "Root 3", "depth": 0, "expanded": False, "has_children": False, "id": "r3"},
        ]
        self.endResetModel()

    @Slot()
    def add_random_item(self):
        # Add a new root at the end
        new_idx = len(self._items)
        self.beginInsertRows(QModelIndex(), new_idx, new_idx)
        self._items.append({
            "name": f"New Root {random.randint(100,999)}",
            "depth": 0,
            "expanded": False,
            "has_children": False,
            "id": f"new_{random.randint(1000,9999)}"
        })
        self.endInsertRows()

    @Slot()
    def remove_random_item(self):
        if not self._items: return
        idx = random.randint(0, len(self._items)-1)
        self.beginRemoveRows(QModelIndex(), idx, idx)
        del self._items[idx]
        self.endRemoveRows()

    @Slot(int)
    def toggle(self, row):
        if row < 0 or row >= len(self._items): return
        
        item = self._items[row]
        if not item["has_children"]: return

        is_expanded = item["expanded"]
        
        # Simulate Expand/Collapse
        if is_expanded:
            # Collapse
            item["expanded"] = False
            self.dataChanged.emit(self.index(row), self.index(row), [self.ExpandedRole])
            
            # Remove children (simplified logic: remove next N items with higher depth)
            count = 0
            depth = item["depth"]
            for i in range(row + 1, len(self._items)):
                if self._items[i]["depth"] > depth:
                    count += 1
                else:
                    break
            
            if count > 0:
                self.beginRemoveRows(QModelIndex(), row + 1, row + count)
                del self._items[row+1 : row+1+count]
                self.endRemoveRows()
        
        else:
            # Expand
            item["expanded"] = True
            self.dataChanged.emit(self.index(row), self.index(row), [self.ExpandedRole])
            
            # Insert fake children
            new_items = [
                {"name": f"Sub {random.randint(1,100)}", "depth": item["depth"]+1, "expanded": False, "has_children": False, "id": "sub"}
            ]
            self.beginInsertRows(QModelIndex(), row + 1, row + len(new_items))
            self._items[row+1:row+1] = new_items
            self.endInsertRows()

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    
    model = MockTreeModel()
    engine.rootContext().setContextProperty("testModel", model)
    
    engine.load("tests/debug_tree.qml")
    
    if not engine.rootObjects():
        sys.exit(-1)
        
    sys.exit(app.exec())
