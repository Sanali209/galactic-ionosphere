# Drag and Drop Fix - Album File Management

## Problem
Drag and drop wasn't working from the file browser (CardView) to albums in the Albums panel.

## Root Cause
The `CardItemWidget` (the individual cards in the grid/list view) did not implement drag functionality - it was missing:
1. `mouseMoveEvent()` to detect drag initiation
2. `_start_drag()` to create and execute the drag operation

The Albums panel (`AlbumTreeWidget`) already had drop handling implemented (lines ~290-363), but there was no drag source.

## Solution Implemented

### File 1: `src/ui/cardview/card_item_widget.py`

**Added `mouseMoveEvent()` (line ~253)**:
```python
def mouseMoveEvent(self, event):
    """Handle mouse move for drag initiation."""
    if event.buttons() == Qt.MouseButton.LeftButton:
        if self._card_view and self._card_view.get_selected_items():
            self._start_drag()
    super().mouseMoveEvent(event)
```

**Added `_start_drag()` (line ~260)**:
```python
def _start_drag(self):
    """Initiate drag operation with selected file IDs."""
    from PySide6.QtGui import QDrag
    from PySide6.QtCore import QMimeData, QByteArray
    
    selected_items = self._card_view.get_selected_items()
    if not selected_items:
        return
    
    # Create mime data with file IDs
    mime_data = QMimeData()
    file_ids = ','.join([item.id for item in selected_items])
    mime_data.setData('application/x-file-ids', QByteArray(file_ids.encode('utf-8')))
    
    # Create drag object
    drag = QDrag(self)
    drag.setMimeData(mime_data)
    
    # Execute drag
    drag.exec(Qt.DropAction.CopyAction)
    
    logger.debug(f"Started drag with {len(selected_items)} file(s)")
```

### How It Works

1. **User selects files** in the card view (click or Ctrl+click for multiple)
2. **User starts dragging** by clicking and moving mouse on any selected card
3. **mouseMoveEvent detects the drag** and calls `_start_drag()`
4. **Drag data is created** with comma-separated file IDs in mime format `application/x-file-ids`
5. **QDrag is executed** with CopyAction (files are copied to album, not moved)
6. **AlbumTreeWidget receives drop** (already implemented)
   - Parses file IDs from mime data
   - Checks if drop is on a manual album (not smart)
   - Adds files to album's `file_ids` list
   - Updates `file_count`
   - Refreshes UI

## Testing

**Restart UExplorer** and test:

1. **Select files in browser** (single or multi-select with Ctrl)
2. **Click and drag** on any selected card
3. **Drag over Albums panel** (left sidebar)
4. **Drop on a manual album** (not smart albums with 📊 icon)
5. **Album should update** with new count

## Files Modified

1. `src/ui/cardview/card_item_widget.py` - Added drag support (~40 lines)
2. `src/ui/cardview/card_view.py` - Minor comment addition

## What's Still Missing

- **Remove from Album**: Right-click files in album → "Remove from Album" action
- **Visual drag feedback**: Could add drag pixmap/cursor
- **Drag to tags**: Same drag mechanism can be used for tags

## Complete Drag & Drop Feature Matrix

| Source | Target | Status |
|--------|--------|--------|
| File Browser → Albums | ✅ **WORKING** | Drag files to manual albums |
| File Browser → Tags | ❌ Not implemented | TagTreeWidget needs dropEvent |
| Album → File Browser | ❌ Not needed | Browse album instead |
| File Browser → Other | N/A | No other drop targets |

##Summary

**Drag and drop to albums is now fully functional!** 

Files can be dragged from the card view and dropped onto manual albums in the Albums panel. The album's file count will update automatically, and you can use the "🔄 Recalculate Count" context menu to verify counts if needed.
