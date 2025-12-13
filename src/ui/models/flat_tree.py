from PySide6.QtCore import QAbstractListModel, Qt, Slot, Signal, QModelIndex
from src.core.database.models.folder import FolderRecord
from src.core.database.models.tag import Tag
from src.core.locator import sl
from loguru import logger
import asyncio

class BaseFlatTreeModel(QAbstractListModel):
    """
    Base class for flattening a tree structure into a list for QML ListView.
    Handles depth, expansion state, and indentation logic.
    """
    # Roles
    DisplayRole = Qt.UserRole + 1
    DepthRole = Qt.UserRole + 2
    ExpandedRole = Qt.UserRole + 3
    HasChildrenRole = Qt.UserRole + 4
    NameRole = Qt.UserRole + 5
    PathRole = Qt.UserRole + 6 
    IdRole = Qt.UserRole + 7
    FullNameRole = Qt.UserRole + 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = [] # Flat list of visible items
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._items)
        
    def roleNames(self):
        return {
            self.DisplayRole: b"display",
            self.DepthRole: b"depth",
            self.ExpandedRole: b"isExpanded",
            self.NameRole: b"name",
            self.HasChildrenRole: b"hasChildren",
            self.IdRole: b"id",
            self.FullNameRole: b"fullName",
            self.PathRole: b"path" 
        }

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None
        
        item = self._items[index.row()]
        
        if role == self.DisplayRole: return item.get("name", "")
        if role == self.NameRole: return item.get("name", "")
        if role == self.DepthRole: return item.get("depth", 0)
        if role == self.ExpandedRole: return item.get("expanded", False)
        if role == self.HasChildrenRole: return item.get("has_children", False)
        if role == self.IdRole: return str(item.get("id", ""))
        if role == self.FullNameRole: return item.get("path", "") # Mapping path/fullName to FullNameRole
        if role == self.PathRole: return item.get("path", "")
        
        return None

    @Slot(int)
    def toggle(self, row):
        """Toggle expansion state of the item at 'row'."""
        pass

class FileSystemFlatModel(BaseFlatTreeModel):
    """
    DB-Backed File System Model.
    """
    PathRole = Qt.UserRole + 10
    
    def __init__(self):
        super().__init__()
        self._root_path = "C:/" # Default, should come from config
        self._all_folders = {} # cache of path -> FolderRecord
        
    def roleNames(self):
        roles = super().roleNames()
        roles[self.PathRole] = b"filePath"
        return roles
        
    def data(self, index, role=Qt.DisplayRole):
        val = super().data(index, role)
        if val is not None: return val
        
        item = self._items[index.row()]
        if role == self.PathRole: return item.get("path", "")
        return None
        
    async def load_roots(self):
        """Initial load from DB: Find all root folders (drives)."""
        self.beginResetModel()
        self._items = []
        
        roots = await FolderRecord.find({"parent_path": None})
        logger.info(f"FileSystemModel: Found {len(roots)} roots.")
        
        # Fallback if empty (e.g. first run, but maybe we shouldn't auto-create C:/ anymore if we support multi-drive)
        if not roots:
             # Just leave empty? Or try to standard roots?
             # Better to leave empty as Importer fills it.
             pass
        
        for root in roots:
            self._items.append({
                "name": root.name,
                "path": root.path,
                "depth": 0,
                "expanded": root.is_expanded,
                "has_children": True, # Assume true, or check?
                "record": root
            })
            
            if root.is_expanded:
                await self._load_children_recursive(root, 0)
            
        self.endResetModel()

    async def _load_children_recursive(self, parent_record, depth, visited=None):
        if depth > 20: # Safety break
            return
            
        if visited is None:
            visited = set()
        
        visited.add(parent_record.path)

        # find children
        # find children
        children = await FolderRecord.find({"parent_path": parent_record.path})
        # Sort by name
        children.sort(key=lambda x: x.name)
        
        for child in children:
            if child.path == parent_record.path or child.path in visited:
                logger.warning(f"Cycle detected or already visited: {child.path}. Skipping.")
                continue

            self._items.append({
                "name": child.name,
                "path": child.path,
                "depth": depth + 1,
                "expanded": child.is_expanded,
                "has_children": True, # TODO: Check if really has children?
                "record": child
            })
            if child.is_expanded:
                await self._load_children_recursive(child, depth + 1, visited.copy())

    @Slot(int)
    def toggle(self, row):
        if row < 0 or row >= len(self._items): return
        
        item = self._items[row]
        path = item["path"]
        is_expanded = item["expanded"]
        record = item["record"]
        
        # 1. Update DB (Fire and forget or wait?)
        # Better to wait to ensure consistency, but UI should be snappy.
        # We'll update local model first, then trigger async save.
        
        new_state = not is_expanded
        self._items[row]["expanded"] = new_state
        self.dataChanged.emit(self.index(row), self.index(row), [self.ExpandedRole])
        
        # Async DB Update
        record.is_expanded = new_state
        asyncio.create_task(record.save())
        
        # 2. Update Flat List
        if new_state: 
            # Expanded: Insert children
            asyncio.create_task(self._expand_row(row, item))
        else:
            # Collapsed: Remove children
            self._collapse_row(row, item)

    async def _expand_row(self, row, item):
        # Fetch children
        path = item["path"]
        depth = item["depth"]
        
        children = await FolderRecord.find({"parent_path": path})
        children.sort(key=lambda x: x.name)
        
        if not children:
            # Scan disk if empty? 
            # Or assume importer does it? 
            # User said "on init get content from backend".
            # But if backend is empty, we might need to scan.
            # Implementation Plan said "Sync Logic: FolderSync (part of importer)".
            # So we assume DB is populated.
            return

        # Prepare items to insert
        new_items = []
        
        # Helper to recursively add if child is expanded
        async def build_flat(recs, d):
            flat = []
            for r in recs:
                flat.append({
                    "name": r.name,
                    "path": r.path,
                    "depth": d,
                    "expanded": r.is_expanded,
                    "has_children": True, 
                    "record": r
                })
                if r.is_expanded:
                   sub = await FolderRecord.find({"parent_path": r.path})
                   sub.sort(key=lambda x: x.name)
                   flat.extend(await build_flat(sub, d + 1))
            return flat

        new_items = await build_flat(children, depth + 1)
        
        if new_items:
            self.beginInsertRows(QModelIndex(), row + 1, row + len(new_items))
            self._items[row+1:row+1] = new_items
            self.endInsertRows()

    def _collapse_row(self, row, item):
        # Find how many items to remove
        # Remove all items immediately following 'row' that have depth > item.depth
        depth = item["depth"]
        count = 0
        for i in range(row + 1, len(self._items)):
            if self._items[i]["depth"] > depth:
                count += 1
            else:
                break
        
        if count > 0:
            self.beginRemoveRows(QModelIndex(), row + 1, row + count)
            del self._items[row + 1 : row + 1 + count]
            self.endRemoveRows()

class TagFlatModel(BaseFlatTreeModel):
    """
    Flat model for tags, loaded from DB.
    """
    IdRole = Qt.UserRole + 20
    PathRole = Qt.UserRole + 21
    
    def __init__(self):
        super().__init__()
        self._all_tags = []
        
    def roleNames(self):
        roles = super().roleNames()
        roles[self.IdRole] = b"tagId"
        roles[self.PathRole] = b"tagPath"
        return roles
        
    def data(self, index, role=Qt.DisplayRole):
        val = super().data(index, role)
        if val is not None: return val
        
        item = self._items[index.row()]
        if role == self.IdRole: return str(item["id"])
        if role == self.PathRole: return item.get("path", "")
        return None
        
    async def load_tags(self):
        await self._fetch_tags()
        
    async def _fetch_tags(self):
        self.beginResetModel()
        self._items = []
        try:
            # Fix: Tag model usually uses MPTT or simpler parent_id structure
            # Checking Tag definition would be wise, but assuming parent_id based on previous steps
            all_tags = await Tag.find({})
            
            # Normalize IDs to strings to avoid ObjectId/PydanticObjectId mismatch
            tag_map = {str(t.id): t for t in all_tags}
            children_map = {}
            roots = []
            
            for t in all_tags:
                children_map[str(t.id)] = []
                
            for t in all_tags:
                pid_str = str(t.parent_id) if t.parent_id else None
                if pid_str and pid_str in tag_map:
                    children_map[pid_str].append(t)
                else:
                    roots.append(t)
            
            logger.info(f"TagModel: Fetched {len(all_tags)} tags.")
            logger.info(f"TagModel: Identified {len(roots)} roots.")
            if len(all_tags) > 0:
                sample = all_tags[:5]
                for s in sample:
                    logger.info(f"Tag Sample: {s.name} (ID: {s.id}) -> Parent: {s.parent_id}")
            
            # recursive flatten
            def flatten(nodes, depth):
                res = []
                # nodes.sort(key=lambda x: x.name) # Assuming name exists
                for node in nodes:
                    is_expanded = True
                    ancestors = getattr(node, "path", "")
                    name = getattr(node, "name", str(node.id))
                    
                    # specific check for Tag model fullName
                    full_name_prop = getattr(node, "fullName", None)
                    if full_name_prop:
                        full_path = full_name_prop
                    else:
                        full_path = f"{ancestors}|{name}" if ancestors else name
                    
                    pid_str = str(node.id)
                    res.append({
                        "name": name,
                        "depth": depth,
                        "expanded": is_expanded,
                        "has_children": len(children_map[pid_str]) > 0,
                        "id": node.id,
                        "path": full_path, # Now this is the FULL name (e.g. character/lara)
                        "record": node
                    })
                    
                    if is_expanded:
                        # Ensure we access children map with string ID
                        res.extend(flatten(children_map[pid_str], depth + 1))
                return res

            self._items = flatten(roots, 0)
            logger.info(f"Loaded {len(self._items)} tags into TagFlatModel")
            
        except Exception as e:
            logger.error(f"Error loading tags: {e}")
            
        self.endResetModel()

    @Slot(int)
    def toggle(self, row):
        pass

