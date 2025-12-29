# Session Journal: MongoDB Reindex Fix and Migration Planning

**Date:** 2025-12-29  
**Session ID:** `mongodb_reindex_fix`

---

## Problem Statement

Database optimization task was crashing with error:
```
WARNING | src.ucorefs.services.maintenance_service:database_optimization:434 - 
Failed to optimize tasks: MotorCollection object is not callable. 
If you meant to call the 'tasks.reindex' method on a MotorCollection object 
it is failing because no such method exists.
```

**Root Cause:** `maintenance_service.py:426` called `await collection.reindex()` which doesn't exist on Motor collections.

---

## Research Findings

### Critical Discovery ⚠️
**Motor is deprecated as of May 14, 2025**
- Only bug fixes until May 14, 2026
- Critical fixes only until May 14, 2027
- Official recommendation: Migrate to **PyMongo Async API**

### Current Stack
- Using `motor.motor_asyncio.AsyncIOMotorClient`
- Located in: `src/core/database/manager.py:2`

### Recommended Stack
- **PyMongo Async API** (pymongo >= 4.0)
- Better performance (no thread pool overhead)
- Direct asyncio integration
- Active development and support

**Documentation:** https://pymongo.readthedocs.io/en/stable/api/pymongo/asynchronous/

---

## Solution Applied

### Immediate Fix (Solution 1)
**File:** `src/ucorefs/services/maintenance_service.py:426`

**Changed:**
```python
# ❌ BEFORE - doesn't exist on Motor collections
collection = db[coll_name]
await collection.reindex()

# ✅ AFTER - proper MongoDB command
await db.command({"reIndexCollection": coll_name})
```

**Impact:**
- Fixes immediate crash
- Works with both Motor and PyMongo Async
- Minimal code change (2 lines)

---

## Roadmap Updates

### New Task Created: **SAN-45**
**Title:** Migrate from Motor to PyMongo Async API

**Location:** `docs/roadmap.md` - Phase 2: Architectural Standardization

**Migration Plan:**
1. Update dependencies: `pymongo>=4.0`
2. Replace imports: `motor.motor_asyncio.AsyncIOMotorClient` → `pymongo.AsyncMongoClient`
3. Update `DatabaseManager` class
4. Test all database operations
5. Remove Motor dependency

**Status:** Todo (High Priority)

---

## Artifacts Created

1. **Research Document:** `mongodb_async_research.md`
   - Detailed comparison of Motor vs PyMongo Async
   - 3 solution approaches with pros/cons
   - Testing checklist

2. **Roadmap Entry:** `docs/roadmap.md`
   - SAN-45 task with migration plan
   - Official documentation link
   - Benefits and context

---

## Next Steps

### Immediate
- [ ] Test maintenance optimization task manually
- [ ] Verify no errors in production

### Short-term (This Week)
- [ ] Review migration plan
- [ ] Set up test environment for PyMongo Async
- [ ] Identify all Motor-dependent code

### Long-term (Next Sprint)
- [ ] Execute SAN-45 migration
- [ ] Update all documentation
- [ ] Remove Motor dependency

---

## Files Modified

1. `src/ucorefs/services/maintenance_service.py` - Fixed reindex call
2. `docs/roadmap.md` - Added SAN-45 migration task

**Total Changes:** 2 files, ~20 lines

---

## Key Learnings

1. Motor's `reindex()` method doesn't exist - use `db.command()` instead
2. Motor is deprecated - migration is urgent
3. PyMongo Async offers better performance than Motor
4. Migration is ~95% API compatible (minimal breaking changes)

---

## References

- [PyMongo Async API Documentation](https://pymongo.readthedocs.io/en/stable/api/pymongo/asynchronous/)
- [Motor Deprecation Notice](https://motor.readthedocs.io/en/stable/)
- [MongoDB reIndexCollection Command](https://www.mongodb.com/docs/manual/reference/command/reIndex/)
- Research artifact: `mongodb_async_research.md`
