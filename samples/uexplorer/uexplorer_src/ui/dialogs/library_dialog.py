"""
Library Settings Dialog

Dialog for adding and managing library roots.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QFileDialog, QMessageBox, QLabel, QListWidgetItem
)
from PySide6.QtCore import Qt
from loguru import logger
from qasync import asyncSlot

# Import FSService
from src.ucorefs import FSService

class LibraryDialog(QDialog):
    """
    Library root management dialog.
    
    Features:
    - List of library roots
    - Add new root
    - Remove root
    - Enable/disable root
    """
    
    def __init__(self, locator, parent=None):
        super().__init__(parent)
        self.locator = locator
        self.fs_service = locator.get_system(FSService)
        
        self.setWindowTitle("Library Settings")
        self.setMinimumSize(600, 400)
        
        self.setup_ui()
        # Load roots using async slot
        self.load_roots()
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Manage Library Roots")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # Root list
        self.root_list = QListWidget()
        self.root_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.root_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Folder...")
        self.add_btn.clicked.connect(self.add_root)
        button_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_root)
        self.remove_btn.setEnabled(False)
        button_layout.addWidget(self.remove_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.root_list.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Apply dark theme
        self.apply_theme()
    
    
    @asyncSlot()
    async def load_roots(self):
        """Load current library roots using asyncSlot."""
        try:
            roots = await self.fs_service.get_roots()
            
            self.root_list.clear()
            for root in roots:
                item = QListWidgetItem(root.path)
                item.setData(Qt.UserRole, str(root._id))
                self.root_list.addItem(item)
            
            logger.debug(f"Loaded {len(roots)} library roots")
        except Exception as e:
            logger.error(f"Failed to load roots: {e}")
    
    
    @asyncSlot()
    async def add_root(self):
        """Add new library root using asyncSlot."""
        # Open directory picker
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Library Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not folder:
            return
        
        try:
            # Add to database via FSService
            root = await self.fs_service.add_library_root(
                folder,
                watch_extensions=["jpg", "jpeg", "png", "gif", "webp", "bmp", "txt", "md"],
                blacklist_paths=[]
            )
            
            # Add to list
            item = QListWidgetItem(root.path)
            item.setData(Qt.UserRole, str(root._id))
            self.root_list.addItem(item)
            
            logger.info(f"Added library root: {folder}")
            
            # Trigger background scan
            try:
                from src.ucorefs.discovery.service import DiscoveryService
                discovery = self.locator.get_system(DiscoveryService)
                if discovery:
                    await discovery.scan_root(root._id, background=True)
                    logger.info(f"Triggered background scan for {folder}")
            except Exception as e:
                logger.error(f"Failed to trigger scan: {e}")
            
        except Exception as e:
            logger.error(f"Failed to add root: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add library root:\n{e}")
    
    def remove_root(self):
        """Remove selected library root."""
        selected = self.root_list.selectedItems()
        if not selected:
            return
        
        # Confirm
        paths = [item.text() for item in selected]
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Remove {len(paths)} library root(s)?\n\n" + "\n".join(paths[:5]),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Schedule async work
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._do_remove_roots(selected))
    
    def _do_remove_roots(self, selected):
        """Actually remove roots synchronously."""
        async def _remove():
            try:
                from bson import ObjectId
                from src.ucorefs.models.directory import DirectoryRecord
                
                for item in selected:
                    root_id = ObjectId(item.data(Qt.UserRole))
                    
                    # Find and delete the root directory record
                    root = await DirectoryRecord.get(root_id)
                    if root:
                        await root.delete()
                    
                    # Remove from list
                    self.root_list.takeItem(self.root_list.row(item))
                
                logger.info(f"Removed {len(selected)} library roots")
                
            except Exception as e:
                logger.error(f"Failed to remove roots: {e}")
                QMessageBox.warning(self, "Error", f"Failed to remove library roots:\n{e}")
        
        # Use qasync's global event loop
        import asyncio
        asyncio.ensure_future(_remove())
    
    def on_selection_changed(self):
        """Handle selection change."""
        has_selection = len(self.root_list.selectedItems()) > 0
        self.remove_btn.setEnabled(has_selection)
    
    def apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
            }
            QListWidget::item:hover {
                background-color: #3d3d3d;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #888888;
            }
        """)
