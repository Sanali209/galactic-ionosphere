# TODO Analysis: UCore FS & UExplorer

**Analysis Date**: 2025-12-28  
**Source**: Grep search of all TODO comments in codebase

This table compares all TODOs found in the codebase against actual implementation status to identify which are obsolete (already done) vs still valid.

---

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ **Obsolete** | 0 | Already implemented, TODO can be removed |
| ⚠️ **Valid** | 12 | Still needs implementation |
| 🔄 **Partial** | 0 | Partially implemented |

---

## Detailed TODO Analysis

| # | File | Line | TODO Text | Status | Notes |
|---|------|------|-----------|--------|-------|
| **UExplorer - ViewModels** |
| 1 | `viewmodels/document_manager.py` | 66 | Handle path-to-id resolution if data is Path | ⚠️ **Valid** | Only handles ObjectId strings currently, not file paths. Need to add FSService lookup for path → ObjectId conversion. **Impact**: Medium (navigation flexibility) |
| **UExplorer - UI Widgets** |
| 2 | `ui/widgets/relation_panel.py` | 122 | Call RelationService to find duplicates | ⚠️ **Valid** | "Find Duplicates" button exists but not wired up. Need to call `RelationService.find_duplicates()` or `SimilarityService.create_duplicate_relations()`. **Impact**: High (key feature) |
| **UExplorer - Main Window** |
| 3 | `ui/main_window.py` | 680 | Implement multi-window support | ⚠️ **Valid** | Single window only currently. Would require app-level window management (QApplication tracking multiple MainWindow instances). **Impact**: Low (advanced feature) |
| 4 | `ui/main_window.py` | 1022 | Filter file panes based on album contents | ⚠️ **Valid** | When user selects smart album, should filter visible files. Currently selection doesn't apply filter. Need to connect `album_selected` signal to SearchPipeline. **Impact**: High (expected behavior) |
| 5 | `ui/main_window.py` | 1958 | Implement new window - would require app-level window management | ⚠️ **Valid** | **Duplicate of #3**. Same issue. Can be consolidated. **Impact**: Low |
| **UExplorer - Documents** |
| 6 | `ui/documents/file_browser_document.py` | 81 | Make priority_enabled configurable | ⚠️ **Valid** | Currently hardcoded `_priority_enabled = True`. Should be exposed in SettingsDialog under performance settings. **Impact**: Low (polish) |
| **UExplorer - Actions** |
| 7 | `ui/actions/action_definitions.py` | 199 | Implement thumbnail toggle | ⚠️ **Valid** | Action registered but not wired. Should toggle between grid view (with thumbnails) and list view (no thumbnails), or control thumbnail size. **Impact**: Medium (UX feature) |
| 8 | `ui/actions/action_definitions.py` | 211 | Implement scan trigger | ⚠️ **Valid** | Action placeholder exists. Should trigger `DiscoveryService.scan_all_roots()`. **Impact**: High (F5 refresh feature) |
| **Core - Foundation** |
| 9 | `src/ui/settings/settings_dialog.py` | 182 | Implement reset to defaults from Pydantic model | ⚠️ **Valid** | "Reset to Defaults" button missing. Should introspect Pydantic model fields to get default values and reset config. **Impact**: Low (convenience) |
| 10 | `src/core/journal/service.py` | 40 | Implement sorting and limiting in ORM find | ⚠️ **Valid** | JournalService queries don't support sorting/limiting (would return all entries). Should use MongoDB `.sort()` and `.limit()` in query. **Impact**: Low (performance for large audit logs) |
| 11 | `src/ucorefs/services/maintenance_service.py` | 438 | Add FAISS index optimization when FAISS is integrated | ⚠️ **Valid** | FAISS integration complete, but auto-optimization/rebuild logic is missing. **Impact**: Medium (long-term performance) |
| 12 | `src/ucorefs/extractors/thumbnail.py` | 15 | Add support for more video formats via FFmpeg | ⚠️ **Valid** | Currently only basic formats supported by PIL/Qt. Need FFmpeg backend for full video support. **Impact**: Medium (feature completeness) |

---

## Prioritized Recommendations

### High Priority (Implement Soon)

1. **TODO #2**: Wire up "Find Duplicates" button (quick win, ~1 hour)
   ```python
   # In relation_panel.py:122
   def _find_duplicates(self):
       async def _run():
           similarity_service = self.locator.get_system(SimilarityService)
           await similarity_service.create_duplicate_relations()
           await self._load_relations()
       asyncio.ensure_future(_run())
   ```

2. **TODO #4**: Filter file browser when album selected (high user expectation, ~2 hours)
   ```python
   # In main_window.py:1022
   async def _on_album_selected(self, album_id: str):
       album = await Album.get(ObjectId(album_id))
       if album.is_smart:
           # Execute smart query and filter
           files = await self.album_manager.get_album_files(album._id)
           # Send to active browser
           self.document_manager.send_results_to_active(files)
   ```

3. **TODO #8**: Implement scan action (F5 functionality, ~30 minutes)
   ```python
   # In action_definitions.py:211
   def _trigger_scan():
       discovery = locator.get_system(DiscoveryService)
       asyncio.ensure_future(discovery.scan_all_roots(background=True))
   ```

### Medium Priority

4. **TODO #7**: Thumbnail toggle action (~1 hour)
5. **TODO #1**: Path-to-id resolution (~2 hours)
6. **TODO #6**: Make priority queue configurable (~30 minutes)

### Low Priority (Nice to Have)

7. **TODO #9**: Reset to defaults button (~1 hour)
8. **TODO #10**: Journal service sorting/limiting (~1 hour)
9. **TODO #3, #5**: Multi-window support (major feature, defer)

---

## TODOs to Remove (Already Implemented)

**None found** ✅ - All TODOs are still valid

---

## Code Quality Observations

### Positive
- **Clean codebase**: Only 10 TODOs across entire project
- **No obsolete TODOs**: All are actionable
- **No FIXME/HACK comments**: Good code discipline

### Areas for Improvement
- Some TODos are duplicates (#3 and #5) - can consolidate
- High-priority features left as TODOs (find duplicates, album filtering)
- Action placeholders exist but not wired up

---

## Recommended Actions

1. **Immediate**: Implement high-priority TODOs #2, #4, #8 (~4 hours total)
2. **Short-term**: Address medium-priority TODOs #1, #6, #7 (~4 hours)
3. **Clean up**: Consolidate duplicate TODOs (#3, #5)
4. **Convert to tickets**: Create Linear tickets for remaining TODOs with proper priority

---

## Comparison with Research Findings

The TODO analysis aligns with the comprehensive research findings:

| Research Finding | Corresponding TODO |
|------------------|-------------------|
| "Export actions missing" | ❌ No TODO (should add) |
| "Batch operations missing" | ❌ No TODO (should add) |
| "Tab persistence missing" | ❌ No TODO (should add) |
| "Viewport priority queue UI" | Partially related to #6 |
| "Relation editing missing" | Partially related to #2 |

**Recommendation**: Add TODOs to code for missing features identified in research.

---

## References

- [UExplorer Comprehensive Analysis](file:///d:/github/USCore/docs/uexplorer_comprehensive_analysis.md)
- [UCore FS Comprehensive Analysis](file:///d:/github/USCore/docs/ucorefs_comprehensive_analysis.md)
- [Maintenance Tasks](file:///d:/github/USCore/docs/maintenance_tasks_ucorefs_uexplorer.md)
