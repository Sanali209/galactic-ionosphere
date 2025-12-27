"""
File and folder selection dialogs.
"""
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QWidget

def select_folder(parent: QWidget = None, 
                 title: str = "Select Folder",
                 start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Show folder selection dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        start_dir: Initial directory
        
    Returns:
        Selected folder path, or None if cancelled
    """
    start_path = str(start_dir) if start_dir else ""
    
    folder = QFileDialog.getExistingDirectory(
        parent,
        title,
        start_path,
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )
    
    if folder:
        return Path(folder)
    return None

def select_file(parent: QWidget = None,
               title: str = "Select File",
               filters: str = "All Files (*.*)",
               start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Show file selection dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        filters: File filters (e.g., "Images (*.png *.jpg);;All Files (*.*)")
        start_dir: Initial directory
        
    Returns:
        Selected file path, or None if cancelled
    """
    start_path = str(start_dir) if start_dir else ""
    
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        title,
        start_path,
        filters
    )
    
    if file_path:
        return Path(file_path)
    return None

def select_files(parent: QWidget = None,
                title: str = "Select Files",
                filters: str = "All Files (*.*)",
                start_dir: Optional[Path] = None) -> list[Path]:
    """
    Show multiple file selection dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        filters: File filters
        start_dir: Initial directory
        
    Returns:
        List of selected file paths, empty if cancelled
    """
    start_path = str(start_dir) if start_dir else ""
    
    file_paths, _ = QFileDialog.getOpenFileNames(
        parent,
        title,
        start_path,
        filters
    )
    
    return [Path(p) for p in file_paths]

def save_file(parent: QWidget = None,
             title: str = "Save File",
             filters: str = "All Files (*.*)",
             start_dir: Optional[Path] = None,
             default_name: str = "") -> Optional[Path]:
    """
    Show file save dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        filters: File filters
        start_dir: Initial directory
        default_name: Default file name
        
    Returns:
        Save file path, or None if cancelled
    """
    start_path = str(start_dir / default_name) if start_dir and default_name else \
                 str(start_dir) if start_dir else default_name
    
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        title,
        start_path,
        filters
    )
    
    if file_path:
        return Path(file_path)
    return None
