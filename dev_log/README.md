# Development Log Index

This directory contains development session notes, research documents, and implementation guides for the UExplorer/USCore project.

---

## Session Summaries

### [Session 2025-12-27: Album & Filter Fixes](session_2025-12-27_album_filters.md)
**Status**: ✅ Complete  
Comprehensive fixes for album/tag filtering system, rating filters, and drag-and-drop functionality. Resolved bidirectional relationship issues and MongoDB query operators.

**Key Fixes**:
- Album include/exclude filters
- Tag exclude filter  
- Drag & drop to albums
- Rating filter (≥ stars)
- Unrated file filter

---

## Album Management

### [Album Management Guide](album_management_guide.md)
Complete guide to album add/remove mechanisms in UExplorer. Documents existing functionality and outlines implementation steps for "Remove from Album" feature.

### [Album Exclude Filter - Fix Plan](album_exclude_filter_fix_plan.md)
**Status**: ✅ Implemented  
Detailed analysis of album exclude filter bug where `$nin` operator was incorrectly used for array fields. Solution: use `$nor` operator instead.

### [Album Exclude Filter - Fixed](album_exclude_filter_fixed.md)
**Status**: ✅ Complete  
Summary of the album/tag exclude filter fix with test cases and examples showing the corrected MongoDB queries.

### [Album Filter Complete Fix](album_filter_complete_fix.md)
**Status**: ✅ Complete  
Complete solution for album include/exclude issues, including the bidirectional relationship fix and rebuild tool usage guide.

### [Album Filters - Final Fix](album_filters_final_fix.md)
**Status**: ✅ Complete  
Final summary of all album filter fixes including drag-and-drop integration and bidirectional relationship updates.

---

## Filter System

### [Include/Exclude Filter Research](filter_include_exclude_research.md)
**Status**: Research  
In-depth analysis of MongoDB query logic for include/exclude filters on array fields. Examines why tags work but albums don't, and proposes debugging strategies.

### [Rating Filter Research](rating_filter_research.md)
**Status**: Research  
Investigation of rating filter issues. Documents the change from exact match to `$gte` (greater-than-or-equal) for proper "X stars or better" behavior.

---

## Drag & Drop

### [Drag and Drop Fix](drag_drop_fix.md)
**Status**: ✅ Implemented  
Implementation of drag-and-drop functionality from file browser (CardView) to albums panel. Documents the addition of `mouseMoveEvent` and `_start_drag` methods.

---

## Index Organization

**By Status**:
- ✅ Complete/Implemented: 6 documents
- 🔬 Research: 2 documents

**By Topic**:
- Album Management: 5 documents
- Filtering: 3 documents  
- Drag & Drop: 1 document
- Session Summaries: 1 document

---

## Quick Reference

**Common Issues**:
- Album filters not working → See [Album Filters - Final Fix](album_filters_final_fix.md)
- Drag and drop not working → See [Drag and Drop Fix](drag_drop_fix.md)
- Rating filter showing nothing → See [Rating Filter Research](rating_filter_research.md)

**Implementation Guides**:
- Adding files to albums → See [Album Management Guide](album_management_guide.md)
- Rebuilding album references → See [Album Filter Complete Fix](album_filter_complete_fix.md)

---

**Last Updated**: 2025-12-27
