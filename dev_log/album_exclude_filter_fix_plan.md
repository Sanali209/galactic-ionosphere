# Album Exclude Filter Issue - Fix Plan

## Problem Statement
When user sets an album to "exclude" (right-click album → Exclude), files from that album are still showing in the file browser.

## Root Cause Analysis

### Current Flow
1. ✅ User right-clicks album → "Exclude from Filter (E)"
2. ✅ `AlbumTreeWidget.exclude_requested` signal emits album_id
3. ✅ `AlbumPanel.toggle_exclude()` adds to `_exclude_albums` set
4. ✅ `AlbumPanel._emit_filter_changed()` emits `filter_changed` signal
5. ✅ `UnifiedQueryBuilder._on_album_filter_changed()` updates `album_exclude` list
6. ✅ `UnifiedSearchQuery.to_mongo_filter()` builds MongoDB filter
7. ❌ **PROBLEM**: Filter logic is incorrect!

### The Bug

**File:** `samples/uexplorer/uexplorer_src/viewmodels/unified_query_builder.py` (lines 96-100)

```python
# Album filters
if self.album_exclude:
    if "album_ids" in mongo:
        mongo["album_ids"]["$nin"] = [ObjectId(a) for a in self.album_exclude]
    else:
        mongo["album_ids"] = {"$nin": [ObjectId(a) for a in self.album_exclude]}
```

**What this does**: "Don't show files where `album_ids` contains any of the excluded albums"

**The problem**:
- Files have `album_ids` as a list (e.g., `[ObjectId("album1"), ObjectId("album2")]`)
- MongoDB `$nin` checks if a value is NOT in a list
- But we're checking if the excluded album IDs are NOT in the `album_ids` array

**Correct query should be**:
```python
# Exclude files that have ANY of the excluded album IDs
{"album_ids": {"$nin": [...]}}  # This ALMOST works...
```

**The real issue**:
`$nin` means "field value is NOT IN this array" but we want "array field does NOT CONTAIN these values"!

We need `$not` + `$in` or just check that the intersection is empty.

### Correct Logic

**What we want**: "Show files where `album_ids` does NOT contain any of the excluded album IDs"

**MongoDB query**:
```python
# Option 1: None of the excluded albums should be in album_ids
{
    "album_ids": {
        "$not": {
            "$elemMatch": {"$in": [ObjectId(a) for a in self.album_exclude]}
        }
    }
}

# Option 2 (simpler): album_ids must not overlap with excluded albums
{
    "album_ids": {
        "$not": {
            "$in": [ObjectId(a) for a in self.album_exclude]
        }
    }
}

# Option 3 (most correct): Use $nor to exclude any file in excluded albums
{
    "$nor": [
        {"album_ids": ObjectId(a)} for a in self.album_exclude
    ]
}
```

**Option 3 is correct** because:
- `album_ids` is an array field on `FileRecord`
- We want to exclude files where `album_ids` contains ANY of the excluded album IDs
- `$nor` with individual album checks ensures no match on any excluded album

## Affected Code

### File: `samples/uexplorer/uexplorer_src/viewmodels/unified_query_builder.py`

**Method**: `UnifiedSearchQuery.to_mongo_filter()` (lines 80-118)

**Current Code** (lines 93-100):
```python
# Album filters
if self.album_include:
    mongo["album_ids"] = {"$all": [ObjectId(a) for a in self.album_include]}
if self.album_exclude:
    if "album_ids" in mongo:
        mongo["album_ids"]["$nin"] = [ObjectId(a) for a in self.album_exclude]
    else:
        mongo["album_ids"] = {"$nin": [ObjectId(a) for a in self.album_exclude]}
```

**Fixed Code**:
```python
# Album filters
if self.album_include:
    # Files must have ALL included albums
    mongo["album_ids"] = {"$all": [ObjectId(a) for a in self.album_include]}

if self.album_exclude:
    # Files must NOT have ANY excluded albums
    # Use $nor to exclude files containing any of these album IDs
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for album_id in self.album_exclude:
        mongo["$nor"].append({"album_ids": ObjectId(album_id)})
```

### Why This Fix Works

1. **`$all`** for include: File must have ALL included albums (intersection)
2. **`$nor`** for exclude: File must NOT match ANY of the conditions
   - Each condition checks if `album_ids` contains one excluded album
   - If file has ANY excluded album, it's filtered out

### Test Cases

After fix, these should work:

| File | Albums | Include Filter | Exclude Filter | Expected Result |
|------|--------|----------------|----------------|-----------------|
| file1.jpg | [A, B] | None | [A] | ❌ HIDDEN (has A) |
| file2.jpg | [B, C] | None | [A] | ✅ SHOWN (no A) |
| file3.jpg | [] | None | [A] | ✅ SHOWN (no albums) |
| file4.jpg | [A, B, C] | [A, B] | None | ✅ SHOWN (has both) |
| file5.jpg | [A] | [A, B] | None | ❌ HIDDEN (missing B) |
| file6.jpg | [A, B] | [A] | [B] | ❌ HIDDEN (has excluded B) |

## Implementation Steps

1. **Fix `to_mongo_filter()` method**
   - Update album exclude logic to use `$nor`
   - Ensure compatibility with existing include logic

2. **Test the fix**
   - Manual testing: Exclude album → verify files hidden
   - Edge cases: Empty albums, multiple excludes, include + exclude

3. **Verify tag and directory filters**
   - Check if they have the same issue
   - Tags use `tag_ids` array (lines 84-91)
   - Current tag logic looks correct (`$all` for include, `$nin` for exclude)
   - **WAIT**: Tag exclude also uses `$nin` - might have same issue!

## Additional Issues Found

### Tag Filters (lines 84-91)

**Current Code**:
```python
if self.tag_exclude:
    if "tag_ids" in mongo:
        mongo["tag_ids"]["$nin"] = [ObjectId(t) for t in self.tag_exclude]
    else:
        mongo["tag_ids"] = {"$nin": [ObjectId(t) for t in self.tag_exclude]}
```

**Same problem!** This won't exclude files with excluded tags correctly.

**Fix**:
```python
if self.tag_exclude:
    # Files must NOT have ANY excluded tags
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for tag_id in self.tag_exclude:
        mongo["$nor"].append({"tag_ids": ObjectId(tag_id)})
```

## Complete Fix

**File**: `samples/uexplorer/uexplorer_src/viewmodels/unified_query_builder.py`

**Replace lines 84-100** with:

```python
# Tag filters
if self.tag_include:
    # Files must have ALL included tags
    mongo["tag_ids"] = {"$all": [ObjectId(t) for t in self.tag_include]}

if self.tag_exclude:
    # Files must NOT have ANY excluded tags
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for tag_id in self.tag_exclude:
        mongo["$nor"].append({"tag_ids": ObjectId(tag_id)})

# Album filters  
if self.album_include:
    # Files must have ALL included albums
    mongo["album_ids"] = {"$all": [ObjectId(a) for a in self.album_include]}

if self.album_exclude:
    # Files must NOT have ANY excluded albums
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for album_id in self.album_exclude:
        mongo["$nor"].append({"album_ids": ObjectId(album_id)})
```

## Summary

- ❌ **Bug**: Album/tag exclude using `$nin` operator incorrectly
- ✅ **Fix**: Use `$nor` with individual element checks
- 📝 **Files to modify**: 1 file, ~20 lines changed
- ⏱️ **Estimated time**: 5 minutes
- 🧪 **Testing**: Manual testing with exclude filters

Ready to implement?
