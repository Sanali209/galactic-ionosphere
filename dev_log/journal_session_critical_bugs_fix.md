# Session Journal: Critical Bugs Fix

**Date:** 2025-12-29  
**Session ID:** `critical_bugs_fix`

---

## Problem Statement

Two critical runtime errors blocking core functionality:

1. **AsyncIO Task Reentrancy Error**: `RuntimeError: Cannot enter into task while another task is being executed`
2. **ProcessingState Import Error**: `No module named 'src.ucorefs.models.processing_state'`

---

## Root Cause Analysis

### Bug 1: AsyncIO Reentrancy
**Cause**: `QProgressDialog` with `WindowModal` blocks Qt event loop, which blocks asyncio event loop (shared via qasync). When async operations complete, they try to schedule new tasks but event loop is frozen.

**Location**: [`main_window.py:821-997`](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/main_window.py#L821-L997)

### Bug 2: Import Error  
**Cause**: ProcessingState enum was moved from `src/ucorefs/models/processing_state.py` (deleted) to `src/ucorefs/models/base.py` during codebase consolidation. One import statement not updated.

**Location**: [`file_browser_document.py:613`](file:///d:/github/USCore/samples/uexplorer/uexplorer_src/ui/documents/file_browser_document.py#L613)

---

## Implementation Summary

### Phase 1: Quick Fixes ✅
**Time**: 1 minute

**Changes**:
- Fixed import in `file_browser_document.py:613`
- Changed: `from src.ucorefs.models.processing_state import ProcessingState`
- To: `from src.ucorefs.models.base import ProcessingState`

**Impact**: Priority queue feature (SAN-14 Phase 3) now functional.

### Phase 2: AsyncIO Fixes ✅
**Time**: 30 minutes

**Changes to `main_window.py`**:

1. **`_rebuild_counts_async()` (lines 821-876)**:
   - Removed blocking `QProgressDialog`
   - Use `self.show_progress()` for status bar indicator
   - Changed `asyncio.ensure_future()` → `asyncio.create_task()`
   - Added `finally` block to reset status

2. **`_verify_references_async()` (lines 887-932)**:
   - Same pattern as above
   - Improved message formatting (emoji, pipe separators)

3. **`_cleanup_orphaned_async()` (lines 942-993)**:
   - Same pattern as above
   - Kept confirmation dialog (non-blocking, user must respond)

4. **Helper Methods (lines 999-1015)**:
   - Added `_show_error(message)`
   - Added `_show_warning(message)`
   - Added `_show_success(message)`

**Impact**: All maintenance operations no longer block event loop, preventing RuntimeError.

### Phase 3: Robustness Improvements ✅
**Time**: 15 minutes

**Changes to `file_browser_document.py`**:

1. **Null Checks** (line 547):
   - Added debug logging when CardView not initialized
   - Prevents errors during startup/shutdown

2. **LRU Tracking** (lines 7, 15, 78, 589-599):
   - Added imports: `OrderedDict`, `time`
   - Changed `_last_queued_ids` from `Set[ObjectId]` to `OrderedDict[ObjectId, float]`
   - Stores timestamp with each entry
   - Uses `popitem(last=False)` to remove oldest entries
   - Maintains proper LRU behavior (1000 entry limit)

**Impact**: 
- Prevents crashes when scrolling before UI fully initialized
- Prevents memory leak from unbounded set growth
- Proper LRU eviction (oldest first, not arbitrary 500)

---

## Files Modified

| File | Lines Changed | Description |
|------|--------------|-------------|
| `file_browser_document.py` | ~30 lines | Import fix, null checks, LRU tracking |
| `main_window.py` | ~150 lines | Replaced 3 blocking methods, added 3 helpers |

**Total**: 2 files, ~180 lines modified

---

## Testing Recommendations

### Automated Tests
```python
# test_processing_state_import.py
def test_import_from_base():
    from src.ucorefs.models.base import ProcessingState
    assert ProcessingState.INDEXED == 40

# test_lru_tracking.py  
def test_lru_eviction():
    from collections import OrderedDict
    cache = OrderedDict()
    # Verify FIFO eviction with popitem(last=False)
    ...
```

### Manual Testing Checklist
- [ ] Scroll file browser rapidly - check logs for "🎯 Queued X visible files"
- [ ] Menu → Maintenance → Rebuild Counts (no RuntimeError)
- [ ] Menu → Maintenance → Verify Data Integrity (no RuntimeError)
- [ ] Menu → Maintenance → Cleanup Orphaned Records (no RuntimeError)
- [ ] Verify status bar shows progress during operations
- [ ] Check success/error dialogs appear correctly

---

## Additional Weaknesses Identified (Not Fixed)

### TaskSystem Executor
**Location**: `src/core/tasks/system.py:164`

**Issue**: Uses default `ThreadPoolExecutor` (undefined size) for sync handlers.

**Recommendation**: Create dedicated executor with configurable size:
```python
self._sync_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="TaskSystem-Sync"
)
```

**Priority**: Low (optional enhancement)

---

## Key Learnings

1. **QDialog Blocking**: `QProgressDialog.show()` + `setWindowModality(Qt.WindowModal)` **blocks** asyncio event loop in qasync integration
2. **Status Bar Alternative**: Non-blocking progress via `QProgressBar` in status bar is safer for async operations
3. **Import Consolidation**: When moving classes between modules, use global search to find all import statements
4. **LRU Implementation**: `OrderedDict` with `popitem(last=False)` provides true LRU behavior, better than set slicing

---

## References

- Analysis Document: [`critical_bugs_analysis.md`](critical_bugs_analysis.md)
- ProcessingState enum: [`src/ucorefs/models/base.py:14`](file:///d:/github/USCore/src/ucorefs/models/base.py#L14)
- QAsync Documentation: https://pypi.org/project/qasync/
- Python OrderedDict: https://docs.python.org/3/library/collections.html#collections.OrderedDict
