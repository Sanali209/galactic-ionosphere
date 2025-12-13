from PySide6.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QTabWidget
from PySide6.QtCore import Qt, Signal
from loguru import logger
import asyncio

class SidebarWidget(QWidget):
    # Signals
    folderSelected = Signal(str, bool) # path, recursive
    tagSelected = Signal(str) # tag_id

    def __init__(self, bridge, fs_model, tag_model, parent=None):
        super().__init__(parent)
        self.fs_model = fs_model 
        self.tag_model = tag_model
        
        # State to prevent concurrent expansion
        self._expanding_items = set()
        
        # Connect bridge signal
        if bridge:
            bridge.galleryRefreshed.connect(self.reload_trees)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self.tabs = QTabWidget()
        
        # 1. Folders Tree
        self.tree_folders = QTreeWidget()
        self.tree_folders.setHeaderLabel("Folders")
        self.tree_folders.itemClicked.connect(self._on_folder_clicked)
        self.tabs.addTab(self.tree_folders, "Folders")
        
        # 2. Tags Tree
        self.tree_tags = QTreeWidget()
        self.tree_tags.setHeaderLabel("Tags")
        self.tree_tags.itemClicked.connect(self._on_tag_clicked)
        self.tabs.addTab(self.tree_tags, "Tags")
        
        self.layout.addWidget(self.tabs)
        
        # Initial Load
        self.reload_trees()

    def reload_trees(self):
        # Trigger async load
        loop = asyncio.get_event_loop()
        loop.create_task(self._load_folders())
        loop.create_task(self._load_tags())

    async def _load_folders(self):
        from src.core.database.models.folder import FolderRecord
        
        # Get roots first (Async)
        roots = await FolderRecord.find({"parent_path": None})
        
        # Then clear and update UI (Sync)
        self.tree_folders.clear()
        
        items = []
        for root in roots:
            item = QTreeWidgetItem([root.name])
            item.setData(0, Qt.UserRole, root.path)
            # Add dummy child to show expansion if needed, or recursive load?
            # For now, recursive load specific depth or on expand.
            # Simple approach: Load everything? No, too slow.
            # Load top level, lazy load children.
            # To lazy load in QTreeWidget, we usually use itemExpanded signal.
            # implementing Lazy Loading:
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            items.append(item)
            
        self.tree_folders.addTopLevelItems(items)
        # Hook expansion
        self.tree_folders.itemExpanded.connect(lambda item: asyncio.create_task(self._on_folder_expand(item)))

    async def _on_folder_expand(self, item):
        path = item.data(0, Qt.UserRole)
        
        # Guard: Already loaded or currently loading
        if item.childCount() > 0 or path in self._expanding_items:
            return 
            
        self._expanding_items.add(path)
        
        try:
            from src.core.database.models.folder import FolderRecord
            
            children = await FolderRecord.find({"parent_path": path})
            children.sort(key=lambda x: x.name)
            
            # Double Check: Did something happen while awaiting?
            if item.childCount() > 0:
                return

            new_items = []
            for child in children:
                c_item = QTreeWidgetItem([child.name])
                c_item.setData(0, Qt.UserRole, child.path)
                c_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                new_items.append(c_item)
                
            item.addChildren(new_items)
            
        except Exception as e:
            logger.error(f"Error expanding folder {path}: {e}")
        finally:
            if path in self._expanding_items:
                self._expanding_items.remove(path)

    def _on_folder_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        self.folderSelected.emit(path, True)

    async def _load_tags(self):
        from src.core.database.models.tag import Tag
        
        all_tags = await Tag.find({})
        
        self.tree_tags.clear()
        
        # Build map
        tag_map = {t.id: t for t in all_tags}
        children_map = {}
        for t in all_tags:
            pid = t.parent_id
            if pid not in children_map: children_map[pid] = []
            children_map[pid].append(t)
            
        # Recursive builder
        def build_branch(parent_id):
            nodes = children_map.get(parent_id, [])
            nodes.sort(key=lambda x: x.name)
            branch_items = []
            for node in nodes:
                item = QTreeWidgetItem([node.name])
                item.setData(0, Qt.UserRole, str(node.id))
                
                children = build_branch(node.id)
                if children:
                    item.addChildren(children)
                branch_items.append(item)
            return branch_items

        roots = build_branch(None)
        self.tree_tags.addTopLevelItems(roots)

    def _on_tag_clicked(self, item, column):
        tag_id = item.data(0, Qt.UserRole)
        self.tagSelected.emit(tag_id)
