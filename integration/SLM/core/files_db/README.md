# Files Database Module

A sophisticated file management and AI-powered content analysis system that provides comprehensive file indexing, metadata extraction, and content understanding capabilities.

## Overview

The files_db module combines traditional file management with advanced AI-powered content analysis, offering:

- **Automatic Background Persistence** - All changes saved immediately without explicit save() calls
- **AI-Powered Content Analysis** - Face detection, embedding generation, content tagging
- **Sophisticated Indexing Pipeline** - Strategy-based processing with multiple AI backends
- **Full Backward Compatibility** - Seamless migration from old fs_db system
- **Component Architecture** - Built on core framework with message bus integration

## Core Components

### Document Models (`odm_models.py`)

#### FileRecord
Represents files with comprehensive metadata and AI analysis results:

```python
# Create file record with auto-save
file_record = FileRecord.new_record(
    local_path="/path/to/file.jpg",
    name="file",
    ext=".jpg",
    size=1024000
)

# All changes auto-persist to database
file_record.description = "Beautiful landscape photo"
file_record.list_append("tags", "nature/landscape")

# Compatibility methods
file_path = file_record.full_path  # Property access
FileRecord.add_file_records_from_folder("/images/")  # Batch indexing
file_record.move_to_folder("/new/location/")  # Move with DB update
```

**Features:**
- Automatic field persistence on change
- List manipulation methods (`list_append`, `list_extend`, `list_remove`)
- Wrapper-style field access (`get_field_val`, `set_field_val`)
- File operations (`move_to_folder`, `refresh_thumb`)
- AI expertise storage (`ai_expertise` field)

#### TagRecord
Hierarchical tag system with relationship management:

```python
# Create hierarchical tags
tag = TagRecord.new_record(
    name="landscape", 
    full_name="nature/landscape",
    parent=parent_tag
)

# Tag file relationships
tag.add_to_file_rec(file_record)
files_with_tag = tag.tagged_files()
child_tags = tag.child_tags()

# Static methods
tags_on_file = TagRecord.get_tags_of_file(file_record)
TagRecord.get_tags_report()  # Generate statistics
```

### Compatibility Models (`compatibility_models.py`)

#### AnnotationJob
Complete annotation workflow management for ML datasets:

```python
# Create annotation job
job = AnnotationJob.new_record(
    name="image_classification",
    choices=["cat", "dog", "bird"]
)

# Annotate files
job.annotate_file(file_record, "cat")
job.add_annotation_choices(["fish", "rabbit"])
job.rename_annotation_label("cat", "feline")

# Get annotations by label
cat_annotations = job.get_ann_records_by_label("cat")
job.clear_job()  # Remove all annotations
```

#### Detection
Object detection results with AI backend tracking:

```python
# Face detection results
detection = Detection.new_record(
    parent_image_id=file_record,
    detection_type="face_detection",
    confidence=95,
    bbox=[100, 150, 200, 250],
    metadata={"backend": "opencv"}
)
```

### Advanced Indexing System (`indexing_system.py`)

Sophisticated AI-powered content analysis pipeline that recreates the capabilities of the old fs_db system.

#### Components

**FileTypeRouter** - Routes files to appropriate AI processing pipelines:
- Image files: Metadata → Face Detection → Embeddings → AI Tagging
- Other files: Basic metadata extraction

**AI Processors:**
- `MetadataProcessor` - EXIF, file stats, timestamps
- `FaceDetectionProcessor` - Multi-backend face detection with object extraction
- `EmbeddingProcessor` - Multiple CNN models (ResNet50, CLIP, DINO, BLIP) with fusion
- `TaggingProcessor` - AI content tagging (DeepDanbooru, SmilingWolf compatible)

#### Usage

```python
# Initialize advanced indexing service
indexing_service = AdvancedIndexingService(message_bus)

# Index single file with full AI processing
await indexing_service.index_file_advanced("files.index_file_advanced", file_record)

# Batch processing with query
query = {'local_path': {'$regex': '^/path/to/images'}}
await indexing_service.index_files_advanced("files.index_advanced", query, max_workers=8)

# Configure indexing behavior
config = update_indexer_config(
    detect_faces=True,
    embedding_models=["resnet50", "clip", "dino"],
    max_workers=6,
    min_detection_size=20
)
```

#### AI Model Integration

The system supports multiple AI backends:

```python
# Face Detection Backends
- opencv (OpenCV DNN face detection)
- mtcnn (Multi-task CNN for face detection)

# Embedding Models  
- resnet50 (ResNet-50 CNN features)
- clip (CLIP vision transformer)
- dino (DINO self-supervised features)
- blip (BLIP multimodal features)

# Content Tagging
- deepdanbooru (Anime/artwork tagging)
- smiling_wolf (General content tagging)
- llava_describe (LLM image description)
```

### Services (`services.py`)

Component-based services for file operations:

#### IndexingService
Basic file discovery and record creation:

```python
# Simple file indexing
await message_bus.publish_async("files.index_file", 
                               file_path="/path/to/file.jpg",
                               metadata={"source": "camera"})
```

#### TagService  
Tag management with hierarchy support:

```python
# Create hierarchical tags
await message_bus.publish_async("tags.create_tag",
                               full_tag_name="nature/landscape/mountain")
```

#### AnnotationService & RelationService
Content annotation and relationship management between documents.

### Compatibility Layer (`compatibility_utils.py`)

**Backward Compatibility Functions:**
```python
# Old fs_db API compatibility
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.files_db.components.fs_tag import TagRecord  
from SLM.files_db.files_functions.index_folder import index_folder
from SLM.files_db.annotation_tool.annotation import AnnotationJob

# All old import paths work seamlessly
files = get_file_record_by_folder("/images/", recurse=True)
refind_exist_files("/images/")  # Remove records for missing files
index_folder_one_thread("/new/images/")  # Single-threaded indexing
annotate_folder("/images/", annotation_job, "positive")
```

**Utility Functions:**
- `get_file_record_by_folder(path, recurse=False)` - Find files in folder
- `refind_exist_files(path)` - Clean up missing file records  
- `remove_files_record_by_mach_pattern(pattern)` - Remove by regex
- `index_folder(query_or_path, workers=4)` - Multi-threaded indexing
- `annotate_folder(path, job, label)` - Batch annotation

## Migration Guide

### From Old fs_db System

The new files_db provides **100% backward compatibility**:

1. **No Code Changes Required** - All old import paths work
2. **Automatic Persistence** - No more manual `save()` calls needed
3. **Enhanced Performance** - Better architecture with automatic optimizations
4. **Extended Features** - New AI capabilities while maintaining old functionality

### Key Differences

```python
# Old way (still works)
file_record = FileRecord(local_path="/path/file.jpg")
file_record.description = "New description"
file_record.save()  # Manual save required

# New way (automatic persistence)
file_record = FileRecord.new_record(local_path="/path/file.jpg")
file_record.description = "New description"  # Auto-saves to database!

# Both approaches work, but new way is recommended
```

## Configuration

### Indexing Configuration

```python
from SLM.core.files_db.indexing_system import IndexingConfig

config = IndexingConfig()
config.detect_faces = True
config.face_detection_backends = ["opencv", "mtcnn"]
config.embedding_models = ["resnet50", "clip", "dino", "blip"]
config.max_workers = 4
config.min_detection_size = 20
```

### Database Integration

The files_db integrates seamlessly with the core mongoODM system:

```python
# Component registration (handled automatically)
from SLM.core.files_db import IndexingService, TagService
from SLM.core.files_db.indexing_system import AdvancedIndexingService

# Services auto-register with dependency injection
```

## Performance Features

### Incremental Processing
- **Backend Tracking** - `backend_indexed` field prevents reprocessing
- **Quality Filtering** - Configurable thresholds for AI detections
- **Batch Operations** - Multi-threaded processing with progress tracking

### Caching System
- **Field Caching** - Optional caching for frequently accessed data
- **Embedding Cache** - Persistent storage for expensive AI computations
- **Database Optimization** - Automatic indexing for common queries

### Scalability
- **Asynchronous Processing** - Non-blocking AI operations
- **Worker Pool Management** - Configurable concurrency control
- **Memory Management** - Optional caching can be disabled

## Event System

The files_db integrates with the core message bus for event-driven operations:

### Events Published
- `files.file_indexed` - Basic file indexing completed
- `files.file_indexed_advanced` - AI processing completed
- `files.batch_indexed_advanced` - Batch processing completed
- `tags.tag_created` - New tag created
- `annotations.created` - New annotation added

### Event Subscriptions
```python
# Subscribe to indexing events
message_bus.subscribe("files.file_indexed_advanced", on_file_processed)

async def on_file_processed(msg_type, file_record, context):
    logger.info(f"Processed {file_record.local_path}")
    if context.get("faces_detected", 0) > 0:
        logger.info(f"Found {context['faces_detected']} faces")
```

## Best Practices

### File Management
1. **Use new_record()** for creating files with auto-save
2. **Leverage list methods** for tag and metadata manipulation
3. **Enable field caching** only when memory allows
4. **Use batch operations** for large-scale processing

### AI Processing  
1. **Configure backends** based on accuracy vs speed requirements
2. **Set appropriate worker counts** based on system resources
3. **Monitor processing progress** with event subscriptions
4. **Use incremental processing** to avoid recomputation

### Performance Optimization
1. **Index frequently queried fields** in MongoDB
2. **Use query-based batch processing** instead of loading all files
3. **Configure appropriate batch sizes** for memory constraints
4. **Enable logging** for debugging and monitoring

## Examples

### Complete File Processing Workflow

```python
from SLM.core.files_db import *

# 1. Discover and index files
FileRecord.add_file_records_from_folder("/images/dataset/")

# 2. Advanced AI processing
indexing_service = AdvancedIndexingService(message_bus)
query = {'indexed_by': {'$nin': ['advanced_indexing']}}
await indexing_service.index_files_advanced("files.index_advanced", query)

# 3. Create annotation workflow
job = AnnotationJob.new_record(
    name="image_classification",
    choices=["positive", "negative", "neutral"]
)

# 4. Annotate files
files = FileRecord.find({'tags': 'object_detect/face'})
for file_record in files:
    job.annotate_file(file_record, "positive")

# 5. Export results
client = SLMAnnotationClient()
client.save_to_json("annotations_export.json")
```

The files_db module provides a complete solution for file management with AI-powered content understanding, combining the best of traditional file systems with modern machine learning capabilities.
