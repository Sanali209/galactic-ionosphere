# Album Management - Add/Remove Files Guide

## What's Already Implemented ✅

### Backend Methods (AlbumManager)

**Add File to Album:**
```python
async def add_file_to_album(album_id: ObjectId, file_id: ObjectId) -> bool
```
- Adds a file to a manual album
- Updates `file_count` automatically
- Returns True if successful

**Remove File from Album:**
```python
async def remove_file_from_album(album_id: ObjectId, file_id: ObjectId) -> bool
```
- Removes a file from a manual album
- Updates `file_count` automatically
- Returns True if successful

**Get Album Files:**
```python
async def get_album_files(album_id: ObjectId, limit: int, offset: int) -> List[FileRecord]
```
- Gets files in an album
- For smart albums: executes query
- For manual albums: retrieves file list
- Supports pagination

---

## UI Features Already Available

### 1. Drag and Drop (IMPLEMENTED ✅)

**How to Add Files to Album:**
1. Select files in file browser
2. Drag them to an album in Albums panel
3. Drop on the album
4. Files are automatically added
5. Album refreshes to show new count

**Implementation Location:**
- File: `samples/uexplorer/uexplorer_src/ui/widgets/album_tree.py`
- Methods: `dragEnterEvent`, `dragMoveEvent`, `dropEvent`
- Lines: ~290-364

**Code:**
```python
def dragEnterEvent(self, event):
    """Accept file drops if dragging over an album."""
    if event.mimeData().hasFormat("application/x-file-ids"):
        event.acceptProposedAction()

def dropEvent(self, event):
    """Handle file drop on album."""
    # Get album from drop position
    item = self.itemAt(event.position().toPoint())
    album_id = item.data(0, Qt.UserRole)
    
    # Get file IDs from drag data
    data = event.mimeData().data("application/x-file-ids")
    file_ids = pickle.loads(data.data())
    
    # Add files to album
    asyncio.ensure_future(self._add_files_to_album_async(album_id, file_ids))
```

---

### 2. Context Menu Remove (TO BE ADDED)

**What's Missing:**
- Right-click file → "Remove from Album" action
- This would appear when viewing files IN an album

**Where to Add:**
File browser context menu needs to know current album context.

---

## How to Add "Remove from Album" Feature

### Step 1: Add Context to File Browser

File browser needs to know when it's displaying an album's contents:

```python
class FileBrowserDocument:
    def __init__(self):
        self.current_album_id = None  # Track current album
    
    def browse_album(self, album_id):
        """Load files from album."""
        self.current_album_id = album_id
        # Load files...
```

### Step 2: Add Context Menu Action

In file browser's context menu:

```python
def _show_context_menu(self, position):
    menu = QMenu(self)
    
    # ... existing actions ...
    
    # Add "Remove from Album" if in album context
    if self.current_album_id:
        menu.addSeparator()
        remove_action = QAction("Remove from Album", self)
        remove_action.triggered.connect(self._remove_from_album)
        menu.addAction(remove_action)
    
    menu.exec_(self.mapToGlobal(position))

def _remove_from_album(self):
    """Remove selected files from current album."""
    selected = self.get_selected_file_ids()
    asyncio.ensure_future(
        self._remove_from_album_async(self.current_album_id, selected)
    )

async def _remove_from_album_async(self, album_id, file_ids):
    """Remove files from album."""
    from src.ucorefs.albums.manager import AlbumManager
    
    album_manager = self.locator.get_system(AlbumManager)
    
    for file_id in file_ids:
        await album_manager.remove_file_from_album(
            ObjectId(album_id),
            ObjectId(file_id)
        )
    
    # Refresh view
    await self.refresh()
```

### Step 3: Connect Album Selection to Browser

When user clicks an album in Albums panel:

```python
# In AlbumTreeWidget._on_item_clicked:
def _on_item_clicked(self, item, column):
    album_id = item.data(0, Qt.UserRole)
    
    # Emit signal with album ID
    self.album_selected.emit(
        album_id,
        is_smart,
        smart_query
    )

# In MainWindow, connect to file browser:
def _on_album_selected(self, album_id, is_smart, query):
    # Tell active browser to show album contents
    active_browser = self.get_active_browser()
    if active_browser:
        active_browser.browse_album(album_id)
```

---

## Quick Implementation Guide

### To Add Remove Functionality:

**1. Modify FileBrowserDocument** (`samples/uexplorer/uexplorer_src/ui/documents/file_browser_document.py`):

```python
# Add to __init__:
self.current_album_id = None

# Add method:
def browse_album(self, album_id: str):
    """Display files from an album."""
    self.current_album_id = album_id
    # Load album files...
```

**2. Add Context Menu in File Browser:**

```python
# In context menu builder:
if self.current_album_id:
    remove_action = QAction("🗑️ Remove from Album", self)
    remove_action.triggered.connect(self._remove_from_album)
    menu.addAction(remove_action)
```

**3. Implement Remove Handler:**

```python
def _remove_from_album(self):
    selected = self.selection_manager.get_selected_ids()
    if selected and self.current_album_id:
        asyncio.ensure_future(
            self._remove_from_album_async(selected)
        )

async def _remove_from_album_async(self, file_ids):
    from src.ucorefs.albums.manager import AlbumManager
    album_manager = self.locator.get_system(AlbumManager)
    
    for fid in file_ids:
        await album_manager.remove_file_from_album(
            ObjectId(self.current_album_id),
            ObjectId(fid)
        )
    
    # Refresh to remove from view
    await self.view_model.refresh()
```

---

## Summary

### ✅ What Works Now:
- **Drag and drop files to albums** - FULLY IMPLEMENTED
- **Backend add/remove methods** - READY TO USE
- **Album file listing** - WORKS

### ⚠️ What Needs Implementation:
- **Remove files from album via context menu**
- **Album browsing in file browser**
- **Visual indication that you're viewing an album**

### 📋 Files to Modify:
1. `samples/uexplorer/uexplorer_src/ui/documents/file_browser_document.py`
   - Add `current_album_id` tracking
   - Add `browse_album()` method
   - Add "Remove from Album" context menu action

2. `samples/uexplorer/uexplorer_src/ui/main_window.py` (optional)
   - Connect album selection to file browser

---

## Estimated Implementation Time: 30 minutes

The backend is complete. Just need to:
1. Track album context in file browser
2. Add one context menu item
3. Wire up the remove handler

Would you like me to implement the "Remove from Album" feature now?
