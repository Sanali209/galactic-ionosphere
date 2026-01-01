# SystemBundle Pattern Implementation - Session Journal

**Date**: 2026-01-01  
**Session ID**: systembundle_implementation  
**Status**: Planning

## Objective

Reduce verbosity of `main.py` by grouping related systems into reusable "Bundles".

## Research Findings

### Current State

**main.py Analysis**:
- 97 lines total
- 18 system registrations via `.add_system()` calls (lines 68-86)
- 16 explicit UCoreFS imports (lines 27-42)
- Manual dependency ordering required

**ApplicationBuilder** (`src/core/bootstrap.py`):
- Fluent builder pattern
- 221 lines
- Methods: `with_default_systems()`, `add_system()`, `with_logging()`, `build()`
- No bundle support currently

### Proposed Architecture

```
SystemBundle (ABC)
├── UCoreFSBundle         # 16 UCoreFS services
└── UExplorerUIBundle     # 2 UI services (SessionState, NavigationService)
```

**Benefits**:
1. Reduced `main.py` from ~18 add_system calls to 2 add_bundle calls
2. Dependency order encapsulated in bundle
3. Reusable across different entry points
4. Testable in isolation

## Design Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | ABC with `register()` method | Simple, matches existing builder pattern |
| D2 | Bundles in domain packages | Keeps related code together (`src/ucorefs/bundle.py`) |
| D3 | Lazy imports in bundles | Avoids circular imports, matches main.py pattern |

## Files to Modify

1. `src/core/bootstrap.py` - Add `SystemBundle` ABC and `add_bundle()` method
2. `src/ucorefs/bundle.py` - NEW: `UCoreFSBundle` implementation
3. `samples/uexplorer/uexplorer_src/bundle.py` - NEW: `UExplorerUIBundle`
4. `samples/uexplorer/main.py` - Refactor to use bundles
5. `tests/core/test_bundle.py` - NEW: Unit tests

## Progress

- [x] Research existing implementation
- [x] Design bundle interface
- [x] Create implementation plan
- [x] User approval
- [x] Implementation
- [x] Verification

---

**Session Start**: 2026-01-01 22:15  
**Session End**: 2026-01-01 22:34  
**Status**: Complete
