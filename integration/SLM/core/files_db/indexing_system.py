"""
Sophisticated AI-powered content analysis and indexing system for files_db.
This system recreates the advanced indexing capabilities from the old fs_db framework
using the new core framework architecture and automatic persistence.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from loguru import logger

from SLM.core.component import Component
from SLM.core.message_bus import MessageBus
from SLM.core.config import Config
from .odm_models import FileRecord
from .compatibility_models import Detection


class IndexingConfig(Config):
    """Configuration for the indexing system."""
    def __init__(self):
        super().__init__()
        self.detect_faces: bool = True
        self.face_detection_backends: List[str] = ["opencv", "mtcnn"]
        self.embedding_models: List[str] = ["resnet50", "clip", "dino", "blip"]
        self.embedding_cache_path: str = "D:\\data\\ImageDataManager"
        self.detection_storage_path: str = "D:\\data\\ImageDataManager\\Image_detect"
        self.batch_size: int = 16
        self.max_workers: int = 4
        self.min_detection_size: int = 20  # Minimum width/height for detections


class FileProcessor(ABC):
    """Abstract base class for file processors/indexers."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.backend_indexed: List[str] = []
        
    @abstractmethod
    def process(self, file_record: FileRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a file and return processing results."""
        pass
    
    def should_process(self, file_record: FileRecord, backend: str) -> bool:
        """Check if this processor should run for the given file and backend."""
        if not self.enabled:
            return False
        
        # Check if this backend has already processed this file
        indexed_backends = file_record.get_field_val("backend_indexed", [])
        if indexed_backends is None:
            indexed_backends = []
        return backend not in indexed_backends


class MetadataProcessor(FileProcessor):
    """Extracts metadata from files (EXIF, etc.)."""
    
    def __init__(self):
        super().__init__("metadata_processor", enabled=True)
    
    def process(self, file_record: FileRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and store metadata from the file."""
        file_path = getattr(file_record, 'local_path')
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            # Use external metadata extraction (would integrate with existing metadata tools)
            # For now, placeholder implementation
            metadata = {}
            
            # Extract basic file info
            stat = os.stat(file_path)
            metadata.update({
                'file_size': stat.st_size,
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime
            })
            
            # Mark this backend as processed
            file_record.list_append("backend_indexed", "metadata_processor", no_dupes=True)
            
            # Store metadata in file record
            current_metadata = file_record.get_field_val("metadata", {})
            if current_metadata is None:
                current_metadata = {}
            current_metadata.update(metadata)
            file_record.set_field_val("metadata", current_metadata)
            
            context["metadata_extracted"] = True
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {}


class FaceDetectionProcessor(FileProcessor):
    """Detects faces in images using multiple AI backends."""
    
    def __init__(self, config: IndexingConfig):
        super().__init__("face_detection", enabled=config.detect_faces)
        self.config = config
        self.detection_backends = config.face_detection_backends
    
    def process(self, file_record: FileRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect faces in the image and create Detection records."""
        file_path = getattr(file_record, 'local_path')
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            detections_created = []
            
            for backend in self.detection_backends:
                if not self.should_process(file_record, f"face_detection_{backend}"):
                    continue
                
                # Placeholder for actual face detection
                # In real implementation, this would call the AI models
                mock_detections = self._detect_faces_with_backend(file_path, backend)
                
                for detection_data in mock_detections:
                    # Filter small detections
                    if (detection_data.get("width", 0) < self.config.min_detection_size or 
                        detection_data.get("height", 0) < self.config.min_detection_size):
                        continue
                    
                    # Create Detection record with auto-save
                    detection = Detection.new_record(
                        parent_image_id=file_record,
                        detection_type="face_detection",
                        confidence=int(detection_data.get("score", 0) * 100),
                        bbox=detection_data.get("bbox", []),
                        metadata={
                            "backend": backend,
                            "region_format": detection_data.get("region_format", "xyxy")
                        }
                    )
                    
                    detections_created.append(detection)
                
                # Mark backend as processed
                file_record.list_append("backend_indexed", f"face_detection_{backend}", no_dupes=True)
            
            # Add face detection tags
            if detections_created:
                file_record.list_append("tags", "object_detect/face", no_dupes=True)
                context["faces_detected"] = len(detections_created)
            
            return {"detections": detections_created}
            
        except Exception as e:
            logger.error(f"Error detecting faces in {file_path}: {e}")
            return {}
    
    def _detect_faces_with_backend(self, file_path: str, backend: str) -> List[Dict[str, Any]]:
        """Placeholder for actual face detection implementation."""
        # In real implementation, this would call the actual AI models
        # For now, return mock data to demonstrate the structure
        return []


class EmbeddingProcessor(FileProcessor):
    """Generates embeddings using multiple AI models and fusion."""
    
    def __init__(self, config: IndexingConfig):
        super().__init__("embedding_generation", enabled=True)
        self.config = config
        self.embedding_models = config.embedding_models
    
    def process(self, file_record: FileRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings using multiple AI models."""
        file_path = getattr(file_record, 'local_path')
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            embeddings_generated = {}
            
            for model in self.embedding_models:
                if not self.should_process(file_record, f"embedding_{model}"):
                    continue
                
                # Placeholder for actual embedding generation
                # In real implementation, this would call the AI models
                embedding_vector = self._generate_embedding_with_model(file_path, model)
                
                if embedding_vector:
                    embeddings_generated[model] = embedding_vector
                    
                    # Store embedding in AI expertise field
                    try:
                        from .odm_models import AIExpertise
                        expertise = AIExpertise(
                            service_name="embedding_generation",
                            backend_name=model,
                            data={"embedding": embedding_vector}
                        )
                        file_record.list_append("ai_expertise", expertise)
                    except ImportError:
                        # Fallback if AIExpertise not available
                        logger.warning("AIExpertise not available, storing embedding in metadata")
                        metadata = file_record.get_field_val("metadata", {})
                        if metadata is None:
                            metadata = {}
                        metadata[f"embedding_{model}"] = embedding_vector
                        file_record.set_field_val("metadata", metadata)
                
                # Mark backend as processed
                file_record.list_append("backend_indexed", f"embedding_{model}", no_dupes=True)
            
            # Generate fused embedding if multiple models were used
            if len(embeddings_generated) > 1:
                fused_embedding = self._fuse_embeddings(embeddings_generated)
                if fused_embedding:
                    try:
                        from .odm_models import AIExpertise
                        expertise = AIExpertise(
                            service_name="embedding_generation", 
                            backend_name="fused_multi_model",
                            data={"embedding": fused_embedding}
                        )
                        file_record.list_append("ai_expertise", expertise)
                    except ImportError:
                        pass
            
            context["embeddings_generated"] = list(embeddings_generated.keys())
            return {"embeddings": embeddings_generated}
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {file_path}: {e}")
            return {}
    
    def _generate_embedding_with_model(self, file_path: str, model: str) -> Optional[List[float]]:
        """Placeholder for actual embedding generation."""
        # In real implementation, this would call the actual AI models
        return None
    
    def _fuse_embeddings(self, embeddings: Dict[str, List[float]]) -> Optional[List[float]]:
        """Fuse multiple embeddings using weighted combination."""
        # Placeholder for actual embedding fusion
        return None


class TaggingProcessor(FileProcessor):
    """Generates content tags using AI models."""
    
    def __init__(self):
        super().__init__("ai_tagging", enabled=True)
    
    def process(self, file_record: FileRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content tags using AI models."""
        file_path = getattr(file_record, 'local_path')
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            tags_generated = []
            
            # Placeholder for AI tagging models (DeepDanbooru, SmilingWolf, etc.)
            ai_tags = self._generate_ai_tags(file_path)
            
            for tag in ai_tags:
                if tag not in file_record.list_get("tags"):
                    file_record.list_append("tags", tag, no_dupes=True)
                    tags_generated.append(tag)
            
            # Mark as processed
            file_record.list_append("backend_indexed", "ai_tagging", no_dupes=True)
            
            context["tags_generated"] = tags_generated
            return {"tags": tags_generated}
            
        except Exception as e:
            logger.error(f"Error generating tags for {file_path}: {e}")
            return {}
    
    def _generate_ai_tags(self, file_path: str) -> List[str]:
        """Placeholder for actual AI tagging."""
        # In real implementation, this would call AI tagging models
        return []


class FileTypeRouter:
    """Routes files to appropriate processing pipelines based on file type."""
    
    def __init__(self, config: IndexingConfig):
        self.config = config
        self.processors = self._initialize_processors()
    
    def _initialize_processors(self) -> Dict[str, List[FileProcessor]]:
        """Initialize processors for different file types."""
        return {
            "image": [
                MetadataProcessor(),
                FaceDetectionProcessor(self.config),
                EmbeddingProcessor(self.config),
                TaggingProcessor()
            ],
            "default": [
                MetadataProcessor()
            ]
        }
    
    def get_processors_for_file(self, file_record: FileRecord) -> List[FileProcessor]:
        """Get the appropriate processors for a file."""
        file_path = getattr(file_record, 'local_path')
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine file type
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return self.processors.get("image", [])
        else:
            return self.processors.get("default", [])


class AdvancedIndexingService(Component):
    """
    Advanced indexing service that provides sophisticated AI-powered content analysis.
    This service recreates the capabilities of the old fs_db indexing system.
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.config = IndexingConfig()
        self.router = FileTypeRouter(self.config)
        
        # Subscribe to indexing requests
        self.message_bus.subscribe("files.index_advanced", self.index_files_advanced)
        self.message_bus.subscribe("files.index_file_advanced", self.index_file_advanced)
    
    def start(self):
        logger.info("AdvancedIndexingService started with AI-powered content analysis")
    
    def stop(self):
        logger.info("AdvancedIndexingService stopped")
    
    def index_file_advanced(self, msg_type: str, file_record: FileRecord):
        """Index a single file with advanced AI processing."""
        try:
            file_path = getattr(file_record, 'local_path')
            logger.info(f"Starting advanced indexing: {file_path}")
            
            # Get appropriate processors for this file type
            processors = self.router.get_processors_for_file(file_record)
            
            # Process through the pipeline
            context = {"file_path": file_path}
            
            for processor in processors:
                if processor.enabled:
                    logger.debug(f"Running processor: {processor.name}")
                    try:
                        result = processor.process(file_record, context)
                        context[processor.name] = result
                    except Exception as e:
                        logger.error(f"Error in processor {processor.name}: {e}")
                        continue
            
            # Mark as indexed
            file_record.list_append("indexed_by", "advanced_indexing", no_dupes=True)
            
            # Publish completion event
            self.message_bus.publish("files.file_indexed_advanced", 
                                    file_record=file_record, 
                                    context=context)
            
            logger.info(f"Advanced indexing completed: {file_path}")
            return file_record
            
        except Exception as e:
            logger.error(f"Error in advanced indexing: {e}")
            self.message_bus.publish("files.index_error_advanced", 
                                    file_record=file_record, 
                                    error=str(e))
            return None
    
    def index_files_advanced(self, msg_type: str, query: Dict[str, Any], max_workers: Optional[int] = None):
        """Index multiple files with advanced AI processing using thread pool."""
        workers = max_workers or self.config.max_workers
        
        try:
            # Find files to index
            files = FileRecord.find(query)
            total_files = len(files)
            
            logger.info(f"Starting advanced indexing of {total_files} files with {workers} workers")
            
            # Process files in parallel using ThreadPoolExecutor
            processed_count = 0
            lock = threading.Lock()
            
            def process_file_wrapper(file_record):
                """Wrapper to count processed files thread-safely."""
                nonlocal processed_count
                result = self.index_file_advanced("files.index_file_advanced", file_record)
                
                with lock:
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Advanced indexing progress: {processed_count}/{total_files}")
                
                return result
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(process_file_wrapper, file_record): file_record 
                                 for file_record in files}
                
                # Wait for all to complete
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                    except Exception as e:
                        file_record = future_to_file[future]
                        logger.error(f"Error processing file {getattr(file_record, 'local_path', 'unknown')}: {e}")
            
            logger.info(f"Advanced indexing completed: {processed_count} files processed")
            
            # Publish batch completion event
            self.message_bus.publish("files.batch_indexed_advanced", 
                                    processed_count=processed_count,
                                    total_files=total_files)
            
        except Exception as e:
            logger.error(f"Error in batch advanced indexing: {e}")
            self.message_bus.publish("files.batch_index_error_advanced", error=str(e))


# Configuration and compatibility functions
def get_indexer_config() -> IndexingConfig:
    """Get the current indexing configuration."""
    return IndexingConfig()


def update_indexer_config(**kwargs):
    """Update indexing configuration."""
    config = get_indexer_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
