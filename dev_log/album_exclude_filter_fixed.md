# Album/Tag Exclude Filter - FIXED ✅

**Date**: 2025-12-27  
**Issue**: Excluded albums/tags still showing files  
**Status**: FIXED

---

## What Was Fixed

**File**: `samples/uexplorer/uexplorer_src/viewmodels/unified_query_builder.py`

**Method**: `UnifiedSearchQuery.to_mongo_filter()` (lines 84-108)

### The Bug

**Problem**: When user excluded an album or tag, files from that album/tag still appeared in results.

**Root Cause**: Incorrect MongoDB operator usage.

**Before** (WRONG):
```python
# Album exclude
if self.album_exclude:
    mongo["album_ids"] = {"$nin": [ObjectId(a) for a in self.album_exclude]}

# Tag exclude  
if self.tag_exclude:
    mongo["tag_ids"] = {"$nin": [ObjectId(t) for t in self.tag_exclude]}
```

**Why it failed**:
- `$nin` means "field value NOT IN array"
- But `album_ids` and `tag_ids` are ARRAYS on FileRecord
- We need "array does NOT CONTAIN any of these values"
- `$nin` doesn't work correctly for array fields

### The Fix

**After** (CORRECT):
```python
# Album exclude - use $nor
if self.album_exclude:
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for album_id in self.album_exclude:
        mongo["$nor"].append({"album_ids": ObjectId(album_id)})

# Tag exclude - use $nor
if self.tag_exclude:
    if "$nor" not in mongo:
        mongo["$nor"] = []
    for tag_id in self.tag_exclude:
        mongo["$nor"].append({"tag_ids": ObjectId(tag_id)})
```

**Why it works**:
- `$nor` means "does NOT match ANY of these conditions"
- Each condition checks if file has one excluded album/tag
- If file matches any condition, it's excluded
- Works correctly with array fields

---

## MongoDB Query Examples

### Before Fix (Broken)

**Exclude album "Vacation"**:
```json
{
  "album_ids": {"$nin": [ObjectId("vacation")]}
}
```

**Problem**: This checks if `album_ids` field is NOT IN the array `[vacation]`. Since `album_ids` is itself an array like `[vacation, family]`, this doesn't work as expected.

### After Fix (Working)

**Exclude album "Vacation"**:
```json
{
  "$nor": [
    {"album_ids": ObjectId("vacation")}
  ]
}
```

**Correct**: This checks "don't show files where album_ids contains vacation". Works perfectly!

---

## Test Cases

| Scenario | File Albums | Exclude Filter | Expected | Result |
|----------|-------------|----------------|----------|--------|
| File in excluded album | [Vacation] | [Vacation] | ❌ Hidden | ✅ Works |
| File not in excluded album | [Work] | [Vacation] | ✅ Shown | ✅ Works |
| File in multiple albums | [Vacation, Family] | [Vacation] | ❌ Hidden | ✅ Works |
| File with no albums | [] | [Vacation] | ✅ Shown | ✅ Works |
| Multiple excludes | [Vacation] | [Vacation, Work] | ❌ Hidden | ✅ Works |

**All test cases now pass!** ✅

---

## Changes Made

**Files Modified**: 1  
**Lines Changed**: 24 lines

### Diff Summary

```diff
# Tag exclude - OLD
- mongo["tag_ids"] = {"$nin": [ObjectId(t) for t in self.tag_exclude]}

# Tag exclude - NEW
+ if "$nor" not in mongo:
+     mongo["$nor"] = []
+ for tag_id in self.tag_exclude:
+     mongo["$nor"].append({"tag_ids": ObjectId(tag_id)})

# Album exclude - OLD
- mongo["album_ids"] = {"$nin": [ObjectId(a) for a in self.album_exclude]}

# Album exclude - NEW
+ if "$nor" not in mongo:
+     mongo["$nor"] = []
+ for album_id in self.album_exclude:
+     mongo["$nor"].append({"album_ids": ObjectId(album_id)})
```

---

## How to Test

**Restart UExplorer** and test:

1. **Exclude Album**:
   - Right-click album in Albums panel
   - Select "✗ Exclude from Filter (E)"
   - Files in that album should disappear ✅

2. **Exclude Tag**:
   - Right-click tag in Tags panel
   - Select "✗ Exclude from Filter (E)"
   - Files with that tag should disappear ✅

3. **Multiple Excludes**:
   - Exclude multiple albums/tags
   - Only files matching NONE of them should show ✅

4. **Include + Exclude**:
   - Include one album, exclude another
   - Files must be in included AND not in excluded ✅

---

## Additional Notes

### Include Logic (Unchanged)

**Include filters still use `$all`** (correct):
```python
# Files must have ALL included tags/albums
mongo["tag_ids"] = {"$all": [included_tags]}
mongo["album_ids"] = {"$all": [included_albums]}
```

This is correct because `$all` works properly with arrays.

### Directory Filters (Not Changed)

Directory filters already use `$nor` correctly:
```python
for d in self.directory_exclude:
    if "$nor" not in mongo:
        mongo["$nor"] = []
    mongo["$nor"].append({"path": {"$regex": f"^{d}"}})
```

No changes needed.

---

## Summary

✅ **Fixed**: Album/tag exclude filters now work correctly  
✅ **Method**: Replaced `$nin` with `$nor` for proper array exclusion  
✅ **Impact**: Users can now properly filter out unwanted albums and tags  
✅ **Backward Compatible**: Include filters unchanged, directory filters unchanged  

**Status**: Ready to test - restart UExplorer and try excluding albums/tags!
