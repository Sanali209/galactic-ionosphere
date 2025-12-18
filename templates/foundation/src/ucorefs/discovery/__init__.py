"""
UCoreFS Discovery Package

Background filesystem discovery and synchronization.
"""
from src.ucorefs.discovery.service import DiscoveryService
from src.ucorefs.discovery.library_manager import LibraryManager
from src.ucorefs.discovery.scanner import DirectoryScanner, ScanResult
from src.ucorefs.discovery.diff import DiffDetector, DiffResult
from src.ucorefs.discovery.sync import SyncManager

__all__ = [
    "DiscoveryService",
    "LibraryManager",
    "DirectoryScanner",
    "ScanResult",
    "DiffDetector",
    "DiffResult",
    "SyncManager",
]
