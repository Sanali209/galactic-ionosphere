"""
Utility functions for compatibility with old fs_db system.
These functions provide the same API as the old fs_db utility functions.
"""

import os
import re
from typing import List, Optional
from tqdm import tqdm
from .odm_models import FileRecord, TagRecord, RelationRecord
from .compatibility_models import Detection, AnnotationJob


def get_file_record_by_folder(folder_path: str, recurse: bool = False) -> List[FileRecord]:
    """
    Get all FileRecord objects for files in a folder.
    Compatibility function matching old fs_db API.
    
    Args:
        folder_path: Path to the folder
        recurse: Whether to search recursively
        
    Returns:
        List of FileRecord objects
    """
    if recurse:
        query = {'local_path': {"$regex": '^' + re.escape(folder_path)}}
    else:
        # For non-recursive, we need to find files directly in the folder
        pattern = '^' + re.escape(folder_path) + r'[/\\][^/\\]+$'
        query = {'local_path': {"$regex": pattern}}
    
    # Filter out None values for type safety
    results = FileRecord.find(query)
    return [record for record in results if record is not None]


def refind_exist_files(folder_path: str):
    """
    Check if files in database records still exist on disk and remove records for missing files.
    Compatibility function matching old fs_db API.
    
    Args:
        folder_path: Path to check files in
    """
    query = {'local_path': {"$regex": '^' + re.escape(folder_path)}}
    records = FileRecord.find(query)
    
    removed_count = 0
    for record in tqdm(records, desc="Checking file existence"):
        if record is not None:
            file_path = getattr(record, 'local_path')
            if not os.path.exists(file_path):
                print(f"Removing record for missing file: {file_path}")
                record.delete()
                removed_count += 1
    
    print(f"Removed {removed_count} records for missing files")


def remove_files_record_by_mach_pattern(pattern: str):
    """
    Remove file records that match a regex pattern.
    Compatibility function matching old fs_db API.
    
    Args:
        pattern: Regex pattern to match against file paths
    """
    query = {'local_path': {"$regex": pattern}}
    records = FileRecord.find(query)
    
    removed_count = 0
    for record in tqdm(records, desc="Removing matching records"):
        if record is not None:
            record.delete()
            removed_count += 1
    
    print(f"Removed {removed_count} records matching pattern: {pattern}")


def index_folder_one_thread(folder_path: str):
    """
    Index files in a folder using single thread.
    Compatibility function matching old fs_db API.
    
    Args:
        folder_path: Path to the folder to index
    """
    print(f"Indexing folder: {folder_path}")
    added_files = FileRecord.add_file_records_from_folder(folder_path)
    print(f"Added {len(added_files)} new file records")
    return added_files


def index_folder(query_or_path, num_threads: int = 1):
    """
    Index files based on query or path with multiple threads.
    Compatibility function matching old fs_db API.
    
    Args:
        query_or_path: MongoDB query dict or folder path string
        num_threads: Number of threads to use (placeholder for compatibility)
    """
    if isinstance(query_or_path, str):
        # It's a path
        return index_folder_one_thread(query_or_path)
    else:
        # It's a query - find files matching query and re-index them
        print(f"Re-indexing files matching query: {query_or_path}")
        records = FileRecord.find(query_or_path)
        
        processed_count = 0
        for record in tqdm(records, desc="Re-indexing files"):
            if record is not None:
                file_path = getattr(record, 'local_path')
                if os.path.exists(file_path):
                    # File exists, ensure it's properly indexed
                    # You could add specific indexing logic here
                    processed_count += 1
                else:
                    # File doesn't exist, remove record
                    print(f"Removing record for missing file: {file_path}")
                    record.delete()
        
        print(f"Processed {processed_count} files for re-indexing")
        return processed_count


def annotate_folder(folder_path: str, annotation_job: AnnotationJob, label: str):
    """
    Annotate all files in a folder with a specific label.
    Compatibility function matching old fs_db API.
    
    Args:
        folder_path: Path to the folder
        annotation_job: AnnotationJob instance
        label: Label to apply to files
    """
    files = get_file_record_by_folder(folder_path, recurse=True)
    
    annotated_count = 0
    for file_record in tqdm(files, desc=f"Annotating with '{label}'"):
        if file_record is not None:
            try:
                annotation_job.annotate_file(file_record, label)
                annotated_count += 1
            except Exception as e:
                print(f"Error annotating {getattr(file_record, 'local_path')}: {e}")
    
    print(f"Annotated {annotated_count} files with label '{label}'")


# Backward compatibility imports - allow old import style
def setup_compatibility_imports():
    """
    Set up imports to match old fs_db structure.
    This allows existing code to import with old paths.
    """
    import sys
    
    # Create mock modules to support old import paths
    if 'SLM.files_db.components.File_record_wraper' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.components.File_record_wraper')
        setattr(mock_module, 'FileRecord', FileRecord)
        setattr(mock_module, 'get_file_record_by_folder', get_file_record_by_folder)
        setattr(mock_module, 'refind_exist_files', refind_exist_files)
        setattr(mock_module, 'remove_files_record_by_mach_pattern', remove_files_record_by_mach_pattern)
        sys.modules['SLM.files_db.components.File_record_wraper'] = mock_module
    
    if 'SLM.files_db.components.fs_tag' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.components.fs_tag')
        setattr(mock_module, 'TagRecord', TagRecord)
        sys.modules['SLM.files_db.components.fs_tag'] = mock_module
    
    if 'SLM.files_db.files_functions.index_folder' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.files_functions.index_folder')
        setattr(mock_module, 'index_folder', index_folder)
        setattr(mock_module, 'index_folder_one_thread', index_folder_one_thread)
        sys.modules['SLM.files_db.files_functions.index_folder'] = mock_module
    
    if 'SLM.files_db.annotation_tool.annotation' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.annotation_tool.annotation')
        from .compatibility_models import AnnotationJob
        setattr(mock_module, 'AnnotationJob', AnnotationJob)
        setattr(mock_module, 'annotate_folder', annotate_folder)
        sys.modules['SLM.files_db.annotation_tool.annotation'] = mock_module
    
    if 'SLM.files_db.object_recognition.object_recognition' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.object_recognition.object_recognition')
        setattr(mock_module, 'Detection', Detection)
        sys.modules['SLM.files_db.object_recognition.object_recognition'] = mock_module
    
    if 'SLM.files_db.components.relations.relation' not in sys.modules:
        import types
        mock_module = types.ModuleType('SLM.files_db.components.relations.relation')
        setattr(mock_module, 'RelationRecord', RelationRecord)
        sys.modules['SLM.files_db.components.relations.relation'] = mock_module


# Call setup function when module is imported
setup_compatibility_imports()


# Additional helper functions
class SLMAnnotationClient:
    """
    Client for managing annotation data.
    Compatibility class matching old fs_db API.
    """
    
    def save_to_json(self, file_path: str):
        """
        Save all annotation data to JSON file.
        Compatibility method matching old fs_db API.
        """
        import json
        from datetime import datetime
        
        # Collect all annotation data
        jobs = AnnotationJob.find({})
        data = {
            'export_date': datetime.utcnow().isoformat(),
            'jobs': []
        }
        
        for job in jobs:
            if job is not None:
                job_data = {
                    'name': getattr(job, 'name'),
                    'description': getattr(job, 'description', ''),
                    'choices': job.list_get('choices'),
                    'annotations': []
                }
                
                # Get all annotation records for this job
                from .compatibility_models import AnnotationJobRecord
                records = AnnotationJobRecord.find({'job': job.pk})
                
                for record in records:
                    if record is not None and record.file is not None:
                        file_path = getattr(record.file, 'local_path')
                        job_data['annotations'].append({
                            'file_path': file_path,
                            'label': getattr(record, 'label'),
                            'created_at': getattr(record, 'created_at').isoformat() if hasattr(record, 'created_at') else None
                        })
                
                data['jobs'].append(job_data)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Annotation data saved to: {file_path}")
