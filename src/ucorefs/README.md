# UCoreFS - Filesystem Database System

A comprehensive MongoDB-backed filesystem database with AI capabilities, built on the Foundation template.

## Overview

UCoreFS replaces traditional filesystem APIs with a powerful database-backed approach, enabling:
- Rich metadata and organization
- AI-powered search and automation
- Complex querying and relations
- Background processing pipelines

## Quick Start

```python
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs import FSService, DiscoveryService

# Build application
app = ApplicationBuilder()
app.register_system(FSService)
app.register_system(DiscoveryService)

# Run
await app.run()

# Use services
fs_service = app.locator.get_system(FSService)
roots = await fs_service.get_roots()
```

## Architecture

### Phase 1: Core Schema
- `FSRecord`, `FileRecord`, `DirectoryRecord` models
- `FSService` with entry points API
- Virtual file support

### Phase 2: Discovery
- `LibraryManager` - watch/blacklist configuration
- `DirectoryScanner` - batch filesystem scanning
- `DiffDetector` - incremental change detection
- `SyncManager` - atomic database updates

### Phase 3: File Types
- `IFileDriver` interface with AI methods
- `FileTypeRegistry` - factory pattern
- Built-in drivers: Image, Text, Default
- XMP metadata extraction with hierarchical tags

### Phase 4: Thumbnails & Search
- `ThumbnailService` - configurable caching
- Hybrid search (metadata + optimized queries)

### Phase 4.5: AI Pipeline
- `SimilarityService` - auto-relation creation
- `LLMService` - batch description generation
- Background task handlers

### Phase 5: Detection & Relations
- `DetectionInstance` - virtual bounding boxes
- Hierarchical `DetectionClass` (MPTT)
- `Relation` system with extensible subtypes

### Phase 6: Tags & Albums
- Hierarchical `Tag` with MPTT
- Synonym/antonym support
- Smart `Album` with dynamic queries

### Phase 7: Rules Engine
- `Rule` with triggers (on_import, on_tag, manual)
- Extensible `ICondition` and `IAction`
- `RulesEngine` for automation

### Phase 8: Query Builder
- Fluent `QueryBuilder` API
- `Q` expressions with AND/OR/NOT
- Vector search integration
- Aggregation pipelines

## Usage Examples

### File Management
```python
# Add library root
root = await fs_service.add_library_root(
    "/photos",
    watch_extensions=["jpg", "png"],
    blacklist_paths=["/photos/.cache"]
)

# Scan for changes
await discovery_service.scan_root(root._id, background=True)
```

### Querying
```python
from src.ucorefs.query import QueryBuilder, Q

# Complex query with logical operators
results = await (QueryBuilder()
    .AND(
        Q.rating_gte(4),
        Q.OR(Q.has_tag(tag1), Q.has_tag(tag2))
    )
    .NOT(Q.extension_in(["tmp"]))
    .order_by("created_at", descending=True)
    .limit(50)
    .execute())
```

### Automation
```python
# Create auto-tagging rule
rule = Rule(
    name="Auto-tag vacation photos",
    trigger="on_import",
    conditions=[
        {"type": "path_contains", "params": {"substring": "vacation"}}
    ],
    actions=[
        {"type": "add_tag", "params": {"tag_id": vacation_tag_id}}
    ]
)
await rule.save()
```

## Testing

```bash
# Run all UCoreFS tests
pytest tests/ucorefs/ -v

# 74 tests across 8 phases
# ~7,200 lines of implementation code
```

## Dependencies

- `motor` → `pymongo>=4.10` - Async MongoDB (migration in progress)
- `pydantic` - Data validation
- `Pillow` - Image processing
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `pyexiv2` - XMP metadata (optional)

## Documentation

See `walkthrough.md` for detailed implementation guide.

## License

MIT
