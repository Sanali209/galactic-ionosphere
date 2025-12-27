"""
UCoreFS - Diff Detector

Detects changes between filesystem and database state.
"""
from typing import List, Dict, Set, Tuple
from datetime import datetime
from loguru import logger

from src.ucorefs.discovery.scanner import ScanResult
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord


class DiffResult:
    """Result of diff detection."""
    
    def __init__(self):
        self.added_files: List[ScanResult] = []
        self.added_dirs: List[ScanResult] = []
        self.modified_files: List[Tuple[FileRecord, ScanResult]] = []
        self.deleted_files: List[FileRecord] = []
        self.deleted_dirs: List[DirectoryRecord] = []
    
    @property
    def total_changes(self) -> int:
        """Total number of changes detected."""
        return (
            len(self.added_files) +
            len(self.added_dirs) +
            len(self.modified_files) +
            len(self.deleted_files) +
            len(self.deleted_dirs)
        )


class DiffDetector:
    """
    Detects differences between filesystem and database.
    
    Compares scan results with database state to identify:
    - New files/directories (added)
    - Modified files (size or mtime changed)
    - Deleted files/directories (in DB but not on disk)
    """
    
    async def detect_changes(
        self,
        scan_results: List[ScanResult],
        root_path: str,
        incremental: bool = False
    ) -> DiffResult:
        """
        Detect changes for a batch of scan results.
        
        Args:
            scan_results: Results from filesystem scan
            root_path: Root directory path
            incremental: If True, only detect additions/modifications (skip deletions)
            
        Returns:
            DiffResult with all detected changes
        """
        diff = DiffResult()
        
        # Build path sets from scan results
        scanned_paths = {r.path for r in scan_results}
        scanned_files = {r.path: r for r in scan_results if not r.is_directory}
        scanned_dirs = {r.path: r for r in scan_results if r.is_directory}
        
        # Query existing records for these paths
        existing_files = await FileRecord.find(
            {"path": {"$in": list(scanned_paths)}}
        )
        existing_dirs = await DirectoryRecord.find(
            {"path": {"$in": list(scanned_paths)}}
        )
        
        # Build existing path sets
        existing_file_paths = {f.path: f for f in existing_files}
        existing_dir_paths = {d.path: d for d in existing_dirs}
        
        # Detect added files
        for path, scan_result in scanned_files.items():
            if path not in existing_file_paths:
                diff.added_files.append(scan_result)
        
        # Detect added directories
        for path, scan_result in scanned_dirs.items():
            if path not in existing_dir_paths:
                diff.added_dirs.append(scan_result)
        
        # Detect modified files (size or mtime changed)
        for path, scan_result in scanned_files.items():
            if path in existing_file_paths:
                existing = existing_file_paths[path]
                
                # Check if modified
                if self._is_modified(existing, scan_result):
                    diff.modified_files.append((existing, scan_result))
        
        # If incremental, we skip deletion detection for this batch
        # because the batch is only a subset of files
        if not incremental:
            await self._detect_deletions_internal(diff, scanned_paths, root_path)
            
        # Log summary
        if diff.total_changes > 0:
            logger.info(
                f"Diff complete: "
                f"+{len(diff.added_files)}F +{len(diff.added_dirs)}D "
                f"~{len(diff.modified_files)}F "
                f"-{len(diff.deleted_files)}F -{len(diff.deleted_dirs)}D"
            )
        
        return diff

    async def detect_deletions(
        self,
        visited_paths: Set[str],
        root_path: str
    ) -> DiffResult:
        """
        Detect deleted files after a full scan.
        
        Args:
            visited_paths: Set of all paths found during scan
            root_path: Root directory path
            
        Returns:
            DiffResult with only deletions
        """
        diff = DiffResult()
        await self._detect_deletions_internal(diff, visited_paths, root_path)
        
        if diff.total_changes > 0:
            logger.info(
                f"Deletion check complete: "
                f"-{len(diff.deleted_files)}F -{len(diff.deleted_dirs)}D"
            )
            
        return diff
        
    async def _detect_deletions_internal(
        self,
        diff: DiffResult,
        scanned_paths: Set[str],
        root_path: str
    ):
        """Internal helper for deletion detection."""
        # Detect deleted files
        # Query all files under this root that weren't in scan
        all_db_files = await FileRecord.find(
            {"path": {"$regex": f"^{root_path}"}}
        )
        
        for db_file in all_db_files:
            if db_file.path not in scanned_paths:
                # File in DB but not on disk = deleted
                diff.deleted_files.append(db_file)
        
        # Detect deleted directories
        all_db_dirs = await DirectoryRecord.find(
            {"path": {"$regex": f"^{root_path}"}}
        )
        
        for db_dir in all_db_dirs:
            if db_dir.path not in scanned_paths and not db_dir.is_root:
                # Directory in DB but not on disk = deleted
                diff.deleted_dirs.append(db_dir)
    
    def _is_modified(
        self,
        existing: FileRecord,
        scan_result: ScanResult
    ) -> bool:
        """
        Check if a file has been modified.
        
        Args:
            existing: Existing FileRecord from database
            scan_result: Current scan result
            
        Returns:
            True if file was modified
        """
        # Check size change
        if existing.size_bytes != scan_result.size:
            return True
        
        # Check modification time
        if existing.modified_at:
            existing_timestamp = existing.modified_at.timestamp()
            # Allow 1 second tolerance for filesystem precision
            if abs(existing_timestamp - scan_result.modified_time) > 1.0:
                return True
        
        return False
