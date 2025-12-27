"""
UCoreFS - Directory Scanner

Scans filesystem directories respecting watch/blacklists.
"""
import os
from typing import Iterator, List, Tuple
from pathlib import Path
from loguru import logger

from src.ucorefs.discovery.library_manager import LibraryManager


class ScanResult:
    """Result of scanning a single item."""
    
    def __init__(
        self,
        path: str,
        is_directory: bool,
        size: int = 0,
        modified_time: float = 0.0
    ):
        # ALWAYS normalize to POSIX (forward slashes)
        self.path = str(Path(path).as_posix())
        self.is_directory = is_directory
        self.size = size
        self.modified_time = modified_time
        self.extension = ""
        
        if not is_directory:
            self.extension = Path(path).suffix.lstrip('.')


class DirectoryScanner:
    """
    Scans filesystem directories in batches.
    
    Respects watch lists (extension filters) and blacklists (excluded paths).
    Yields results in batches to avoid memory issues with massive collections.
    """
    
    def __init__(self, library_manager: LibraryManager, batch_size: int = 1000):
        """
        Initialize directory scanner.
        
        Args:
            library_manager: LibraryManager for checking filters
            batch_size: Number of items per batch
        """
        self.library_manager = library_manager
        self.batch_size = batch_size
    
    def scan_directory(
        self,
        root_path: str,
        watch_extensions: List[str] = None,
        blacklist_paths: List[str] = None,
        recursive: bool = True
    ) -> Iterator[List[ScanResult]]:
        """
        Scan a directory and yield results in batches.
        
        Args:
            root_path: Root directory to scan
            watch_extensions: File extensions to include
            blacklist_paths: Paths to exclude
            recursive: Whether to scan subdirectories
            
        Yields:
            List of ScanResult (variable batch size)
        """
        dir_batch = []
        file_batch = []
        
        try:
            for item in self._walk_directory(
                root_path,
                watch_extensions or [],
                blacklist_paths or [],
                recursive
            ):
                if item.is_directory:
                    dir_batch.append(item)
                    # Yield directories frequently (e.g., every 50) to popluate tree fast but avoid UI freeze
                    if len(dir_batch) >= 50:
                        yield dir_batch
                        dir_batch = []
                else:
                    file_batch.append(item)
                    # Yield files in larger chunks for performance
                    if len(file_batch) >= self.batch_size:
                        yield file_batch
                        file_batch = []
                
                # If we have a mix and one is ready, we could yield? 
                # Be careful not to yield just 1 file if batch is 1000.
                
            # Yield remaining items
            if dir_batch:
                yield dir_batch
            if file_batch:
                yield file_batch
                
        except Exception as e:
            logger.error(f"Error scanning {root_path}: {e}")
            if dir_batch:
                yield dir_batch
            if file_batch:
                yield file_batch
    
    def _walk_directory(
        self,
        root_path: str,
        watch_extensions: List[str],
        blacklist_paths: List[str],
        recursive: bool
    ) -> Iterator[ScanResult]:
        """
        Walk directory tree and yield individual results.
        
        Args:
            root_path: Root directory
            watch_extensions: Extensions to include
            blacklist_paths: Paths to exclude
            recursive: Scan subdirectories
            
        Yields:
            Individual ScanResult items
        """
        if not os.path.exists(root_path):
            logger.warning(f"Path does not exist: {root_path}")
            return
        
        try:
            with os.scandir(root_path) as entries:
                for entry in entries:
                    try:
                        # Check blacklist first
                        if self.library_manager.is_blacklisted(
                            entry.path,
                            blacklist_paths
                        ):
                            continue
                        
                        # Get stats
                        stat = entry.stat(follow_symlinks=False)
                        
                        if entry.is_dir(follow_symlinks=False):
                            # Yield directory
                            yield ScanResult(
                                path=entry.path,
                                is_directory=True,
                                modified_time=stat.st_mtime
                            )
                            
                            # Recurse into subdirectory
                            if recursive:
                                yield from self._walk_directory(
                                    entry.path,
                                    watch_extensions,
                                    blacklist_paths,
                                    recursive
                                )
                        
                        elif entry.is_file(follow_symlinks=False):
                            # Check extension whitelist
                            extension = Path(entry.name).suffix.lstrip('.')
                            
                            if self.library_manager.should_scan_extension(
                                extension,
                                watch_extensions
                            ):
                                yield ScanResult(
                                    path=entry.path,
                                    is_directory=False,
                                    size=stat.st_size,
                                    modified_time=stat.st_mtime
                                )
                    
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Cannot access {entry.path}: {e}")
                        continue
        
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot scan directory {root_path}: {e}")
    
    async def scan_all_roots(self) -> Iterator[Tuple[str, List[ScanResult]]]:
        """
        Scan all enabled library roots.
        
        Yields:
            Tuples of (root_path, batch_of_results)
        """
        roots = await self.library_manager.get_enabled_roots()
        
        for root in roots:
            logger.info(f"Scanning library root: {root.path}")
            
            async for batch in self.scan_directory_async(
                root.path,
                root.watch_extensions,
                root.blacklist_paths,
                recursive=True
            ):
                yield (root.path, batch)
    
    async def scan_directory_async(
        self,
        root_path: str,
        watch_extensions: List[str] = None,
        blacklist_paths: List[str] = None,
        recursive: bool = True
    ):
        """
        Async version of scan_directory that runs blocking I/O in a thread pool.
        
        This prevents UI freezing during large directory scans by offloading
        the synchronous os.scandir calls to a background thread.
        
        Args:
            root_path: Root directory to scan
            watch_extensions: File extensions to include
            blacklist_paths: Paths to exclude
            recursive: Whether to scan subdirectories
            
        Yields:
            List of ScanResult (variable batch size)
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        
        # Use a thread pool for blocking I/O
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scanner") as executor:
            # Collect all results in thread, then yield batches
            def collect_results():
                """Run in thread: collect all scan results."""
                return list(self.scan_directory(
                    root_path,
                    watch_extensions,
                    blacklist_paths,
                    recursive
                ))
            
            # Run blocking scan in thread
            batches = await loop.run_in_executor(executor, collect_results)
            
            # Yield batches back to async context
            for batch in batches:
                yield batch
                # Allow event loop to process other events between batches
                await asyncio.sleep(0)

