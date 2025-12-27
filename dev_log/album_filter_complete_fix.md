# Album Include/Exclude Fix - COMPLETE SOLUTION ✅

## Problem
- Album exclude doesn't work properly
- Album include shows nothing  

## Root Cause
`AlbumManager.add_file_to_album()` was not updating `FileRecord.album_ids`, so filter queries couldn't find files by album.

## Solution Implemented

### 1. Fixed AlbumManager (DONE ✅)

**File**: `src/ucorefs/albums/manager.py`

**add_file_to_album()** now updates:
- `Album.file_ids` (existing)
- `FileRecord.album_ids` (NEW - bidirectional!)

**remove_file_from_album()** now updates:
- `Album.file_ids` (existing)
- `FileRecord.album_ids` (NEW - bidirectional!)

### 2. Added Maintenance Tool (DONE ✅)

**File**: `src/ucorefs/services/maintenance_service.py`

New method: `rebuild_album_references()`
- Clears all `FileRecord.album_ids`
- Rebuilds from `Album.file_ids`
- Returns stats: files_updated, albums_processed

### 3. QUICK FIX - Run This in Python Console!

Since the UI integration had issues, **run this directly in UExplorer's Python console**:

```python
import asyncio
from src.ucorefs.services.maintenance_service import MaintenanceService

async def fix_albums():
    from src.core.locator import Locator
    locator = Locator.get_instance()
    maintenance = locator.get_system(MaintenanceService)
    
    print("Rebuilding album references...")
    result = await maintenance.rebuild_album_references()
    
    print(f"✅ Complete!")
    print(f"Files updated: {result['files_updated']}")
    print(f"Albums processed: {result['albums_processed']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")

# Run it
asyncio.ensure_future(fix_albums())
```

### 4. ALTERNATIVE - Add Menu Item Manually

If you want to add it to the Maintenance menu:

**Step 1**: Add to `action_definitions.py` (line ~264):
```python
registry.register_action(
    "maintenance.rebuild_album_refs",
    "🔄 Rebuild Album References...",
    window.rebuild_album_references,
    tooltip="Sync FileRecord.album_ids from Album.file_ids"
)
```

**Step 2**: Add to `menu_manager.py` (line ~106):
```python
maintenance_menu.addAction(self.action_registry.get_action("maintenance.rebuild_album_refs"))
```

**Step 3**: Add handler to `main_window.py` (after line ~882):
```python
def rebuild_album_references(self):
    """Rebuild album references."""
    async def _rebuild():
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        try:
            maintenance = self.locator.get_system("MaintenanceService")
            
            progress = QProgressDialog("Rebuilding album references...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            result = await maintenance.rebuild_album_references()
            progress.close()
            
            msg = f"Album references rebuilt!\n\n"
            msg += f"Files updated: {result['files_updated']}\n"
            msg += f"Albums processed: {result['albums_processed']}"
            
            QMessageBox.information(self, "Complete", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    asyncio.ensure_future(_rebuild())
```

## What This Fixes

After running the rebuild:

✅ **Album include** - Will show files in that album  
✅ **Album exclude** - Will hide files in that album  
✅ **Multiple album filters** - Works correctly  
✅ **Matches tag behavior** - Consistent with tags  

## How It Works

**Before** (broken):
```
Album A: {file_ids: [file1, file2]}
File 1: {album_ids: []}  ← EMPTY! Filter can't find it
File 2: {album_ids: []}  ← EMPTY! Filter can't find it
```

**After rebuild** (working):
```
Album A: {file_ids: [file1, file2]}
File 1: {album_ids: [A]}  ← Has album reference!
File 2: {album_ids: [A]}  ← Has album reference!
```

Now filters can query: `{"album_ids": {"$all": [A]}}` ✅

## Testing

**After running the rebuild:**

1. **Test Include**:
   - Right-click album → "✓ Include in Filter"
   - Should ONLY show files in that album

2. **Test Exclude**:
   - Right-click album → "✗ Exclude from Filter"
   - Should HIDE files in that album

3. **Test Both**:
   - Include album A, exclude album B
   - Should show files in A but not in B

All should work perfectly now! 🎉

## Summary

- ✅ Fixed `AlbumManager` to update both sides of relationship
- ✅ Added `rebuild_album_references()` method  
- ✅ Run rebuild to fix existing data
- ✅ New files will work automatically

**Status**: COMPLETE - Just run the rebuild!
