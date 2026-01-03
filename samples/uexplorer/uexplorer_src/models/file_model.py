"""
File Model for UExplorer

QAbstractItemModel implementation for browsing filesystem through FSService.
"""
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Set
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal, QObject, QTimer
from PySide6.QtGui import QIcon, QPixmap
from bson import ObjectId
from loguru import logger

# Import FSService and ThumbnailService
from src.ucorefs import FSService
from src.ucorefs.thumbnails.service import ThumbnailService

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator

class FileModel(QAbstractItemModel):
    """
    TODO: need resarch is used or not
    Qt model for filesystem browsing.
    
    Features:
    - Lazy loading of directory contents
    - Caching for performance
    - Icons based on file type
    - Sorting support
    """
    
    # Custom roles
    PathRole = Qt.UserRole + 1
    IdRole = Qt.UserRole + 2
    IsDirectoryRole = Qt.UserRole + 3
    SizeRole = Qt.UserRole + 4
    
    def __init__(self, locator: "ServiceLocator") -> None:
        """
        Initialize file model.
        
        Args:
            locator: ServiceLocator instance
        """
        super().__init__()
        
        self.locator: "ServiceLocator" = locator
        self.fs_service: FSService = locator.get_system(FSService)
        self.thumbnail_service: ThumbnailService = locator.get_system(ThumbnailService)
        
        # Cache of loaded records
        self._roots: List[Any] = []  # List of root records
        self._cache: Dict[str, Any] = {}  # record_id -> record
        self._children: Dict[str, List[Any]] = {}  # parent_id -> list of children
        self._thumbnail_cache: Dict[str, QIcon] = {} # record_id -> QIcon
        self._loading_thumbnails: Set[str] = set() # record_id
        
        # Drag state - pause thumbnail loading during drag to prevent UI freezes
        self._drag_in_progress = False
        
        # Virtualization state
        self._children_offset = {}  # parent_id -> number of children loaded
        self._has_more = {}  # parent_id -> True if more children available
        self.FETCH_SIZE = 100  # Items to load per fetch
        
        # Qt requires integer pointers for createIndex, so we map strings to ints
        self._id_to_int = {}  # record_id_str -> int
        self._int_to_id = {}  # int -> record_id_str
        self._next_int_id = 1
        
        # COMPLETELY DISABLED AUTO-LOAD for qasync testing
        # Even direct asyncio.ensure_future() causes qasync loop to exit!
        # Roots must be loaded manually via refresh_roots()
        #
        # asyncio.ensure_future(self._load_roots())
        
        # Subscribe to realtime filesystem updates
        # Subscribe to realtime filesystem updates
        try:
            from src.core.commands.bus import CommandBus
            bus = self.locator.get_system(CommandBus)
            if hasattr(bus, 'subscribe'):
                bus.subscribe("filesystem.updated", self._on_filesystem_updated)
        except (KeyError, ImportError, AttributeError):
            # CommandBus not available or old version
            pass

        # Update throttling
        self._pending_updates = False
        self._update_timer = QTimer()
        self._update_timer.setInterval(20000) # 20 seconds debounce per user request
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._process_pending_updates)

    async def _on_filesystem_updated(self, event: dict):
        """Handle realtime filesystem updates."""
        try:
            # Mark update as pending
            self._pending_updates = True
            
            # Start timer if not running
            if not self._update_timer.isActive():
                # If this is the first update in a while, maybe process immediately?
                # No, better to buffer to avoid freeze storms.
                self._update_timer.start()
            
        except Exception as e:
            logger.error(f"Error handling filesystem update: {e}")
            
    def _process_pending_updates(self):
        """Process buffered updates on main thread."""
        if self._pending_updates:
            logger.debug("Processing pending filesystem updates...")
            
            # Use ResetModel to ensure safety. 
            # Clearing _children breaks parent() lookups for existing indexes, 
            # which crashes QTreeView if we use layoutChanged.
            self.beginResetModel()
            
            # Clear children cache to force reload from DB
            # We preserve _roots to keep the top level stable? 
            # Actually, resetModel invalidates everything anyway.
            self._children = {}
            
            self.endResetModel()
            
            self._pending_updates = False
    
    def _register_id(self, record_id_str: str) -> int:
        """Register a record ID and return its integer representation."""
        if record_id_str not in self._id_to_int:
            int_id = self._next_int_id
            self._next_int_id += 1
            self._id_to_int[record_id_str] = int_id
            self._int_to_id[int_id] = record_id_str
            return int_id
        return self._id_to_int[record_id_str]
    
    async def refresh_roots(self):
        """Refresh library roots - call from main thread as async method."""
        from loguru import logger
        logger.debug("FileModel.refresh_roots() called")
        
        # Clear cache to force re-fetch of children
        self.clear_cache()
        
        # Directly await the load (works safely in qasync loop)
        await self._load_roots()
        
    def clear_cache(self):
        """Clear all cached data."""
        self.beginResetModel()
        self._roots = []
        self._cache = {}
        self._children = {}
        self._thumbnail_cache = {}
        self._previous_thumbnails_loading = self._loading_thumbnails
        self._loading_thumbnails = set()
        
        # Keep ID mappings to avoid invalidating existing QModelIndexes if possible?
        # Actually, resetModel invalidates all indexes, so we should reset IDs too 
        # to prevent memory growth, BUT we must be careful if views hold persistent indexes.
        # ResetModel tells views to discard everything.
        self._id_to_int = {}
        self._int_to_id = {}
        self._next_int_id = 1
        self.endResetModel()

    async def _load_roots(self):
        """Load library roots."""
        try:
            logger.debug("_load_roots: Starting...")
            roots = await self.fs_service.get_roots()
            logger.debug(f"_load_roots: Got {len(roots)} roots from service")
            
            self._roots = roots
            
            # Cache roots
            for root in roots:
                self._cache[str(root._id)] = root
                logger.debug(f"_load_roots: Cached root {root._id}: {root.path}")
                
            # Notify view
            logger.debug("_load_roots: Emitting layoutChanged signal")
            self.layoutChanged.emit()
            
            logger.debug(f"Loaded {len(roots)} library roots")
            
        except Exception as e:
            logger.exception(f"FATAL ERROR in _load_roots: {e}")
            import traceback
            traceback.print_exc()
    
    def set_files(self, files: list):
        """
        Set files to display from external source (ViewModel).
        
        This method allows the tree/list views to display search results
        from BrowseViewModel instead of auto-loading from FSService.
        
        Args:
            files: List of FileRecord objects to display as flat list
        """
        self.beginResetModel()
        
        # Clear all caches
        self._roots = []
        self._cache = {}
        self._children = {}
        self._id_to_int = {}
        self._int_to_id = {}
        self._next_int_id = 1
        
        # Use files as roots (flat display)
        self._roots = files
        
        # Cache all files
        for file in files:
            self._cache[str(file._id)] = file
        
        self.endResetModel()
        logger.debug(f"FileModel.set_files: Displaying {len(files)} files")
    
    def set_sort(self, field: str, ascending: bool = True):
        """
        Sort currently displayed files.
        
        Args:
            field: Field to sort by (name, size, modified_at, rating)
            ascending: True for ascending, False for descending
        """
        if not self._roots:
            return
        
        # Define sort key functions
        def get_sort_key(record):
            if field == "name":
                return getattr(record, 'name', '').lower()
            elif field == "size":
                return getattr(record, 'size_bytes', 0)
            elif field == "modified" or field == "modified_at":
                return getattr(record, 'modified_at', None) or getattr(record, 'created_at', None)
            elif field == "rating":
                return getattr(record, 'rating', 0)
            elif field == "extension":
                return getattr(record, 'extension', '').lower()
            elif field == "file_type":
                return getattr(record, 'file_type', '')
            else:
                return getattr(record, 'name', '').lower()
        
        self.beginResetModel()
        self._roots = sorted(self._roots, key=get_sort_key, reverse=not ascending)
        self.endResetModel()
        
        logger.debug(f"FileModel sorted by {field} {'asc' if ascending else 'desc'}")
    
    def set_group(self, group_by: str = None):
        """
        Group files by field.
        
        Note: Groups are handled as sort+visual separators. Files are sorted by
        group field first, then by name within groups.
        
        Args:
            group_by: Field to group by (file_type, rating, date) or None
        """
        self._group_by = group_by
        
        if not self._roots or not group_by:
            return
        
        # Sort by group field first, then by name
        def get_group_key(record):
            if group_by == "file_type":
                return (getattr(record, 'file_type', 'other'), getattr(record, 'name', '').lower())
            elif group_by == "rating":
                return (-getattr(record, 'rating', 0), getattr(record, 'name', '').lower())
            elif group_by == "date":
                dt = getattr(record, 'modified_at', None) or getattr(record, 'created_at', None)
                date_str = dt.strftime('%Y-%m-%d') if dt else '1970-01-01'
                return (date_str, getattr(record, 'name', '').lower())
            else:
                return (getattr(record, 'name', '').lower(),)
        
        self.beginResetModel()
        self._roots = sorted(self._roots, key=get_group_key)
        self.endResetModel()
        
        logger.debug(f"FileModel grouped by {group_by}")
    
    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        """Create index for item."""
        # logger.debug(f"index() called: row={row}, column={column}, parent_valid={parent.isValid()}")
        try:
            if not self.hasIndex(row, column, parent):
                return QModelIndex()
            
            if not parent.isValid():
                # Root level
                if row < len(self._roots):
                    record = self._roots[row]
                    record_id_str = str(record._id)
                    int_id = self._register_id(record_id_str)
                    return self.createIndex(row, column, int_id)
            else:
                # Child level
                parent_ptr = parent.internalId()
                parent_id_str = self._int_to_id.get(parent_ptr)
                if not parent_id_str:
                    logger.warning(f"index(): Parent ID {parent_ptr} not found in mapping")
                    return QModelIndex()
                    
                children = self._children.get(parent_id_str, [])
                if row < len(children):
                    record = children[row]
                    record_id_str = str(record._id)
                    int_id = self._register_id(record_id_str)
                    return self.createIndex(row, column, int_id)
            
            return QModelIndex()
        except Exception as e:
            logger.exception(f"Error in index({row}, {column}): {e}")
            return QModelIndex()
    
    def parent(self, index: QModelIndex) -> QModelIndex:
        """Get parent index."""
        try:
            if not index.isValid():
                return QModelIndex()
            
            child_id = index.internalId()
            record_id_str = self._int_to_id.get(child_id)
            if not record_id_str:
                return QModelIndex()
                
            record = self._cache.get(record_id_str)
            
            if not record or not hasattr(record, 'parent_id') or not record.parent_id:
                return QModelIndex()
            
            parent_id = str(record.parent_id)
            parent_record = self._cache.get(parent_id)
            
            if not parent_record:
                return QModelIndex()
            
            # Find parent's row
            if not hasattr(parent_record, 'parent_id') or not parent_record.parent_id:
                # Parent is root
                row = next((i for i, r in enumerate(self._roots) if str(r._id) == parent_id), -1)
                if row >= 0:
                    int_id = self._register_id(parent_id)
                    return self.createIndex(row, 0, int_id)
            else:
                # Parent has grandparent
                grandparent_id = str(parent_record.parent_id)
                siblings = self._children.get(grandparent_id, [])
                row = next((i for i, r in enumerate(siblings) if str(r._id) == parent_id), -1)
                if row >= 0:
                    int_id = self._register_id(parent_id)
                    return self.createIndex(row, 0, int_id)
            
            return QModelIndex()
        except Exception as e:
            logger.exception(f"Error in parent(): {e}")
            return QModelIndex()
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Get row count."""
        try:
            if not parent.isValid():
                return len(self._roots)
            
            parent_ptr = parent.internalId()
            parent_id_str = self._int_to_id.get(parent_ptr)
            if not parent_id_str:
                return 0
            
            # Check if we've loaded children
            if parent_id_str not in self._children:
                # Trigger async load (use ensure_future for qasync)
                asyncio.ensure_future(self._load_children(parent_id_str))
                return 0
            
            return len(self._children.get(parent_id_str, []))
        except Exception as e:
            logger.exception(f"Error in rowCount(): {e}")
            return 0
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Get column count."""
        return 4  # Name, Size, Type, Modified
    
    def canFetchMore(self, parent: QModelIndex) -> bool:
        """Check if more children can be fetched (virtualization)."""
        if not parent.isValid():
            return False
        
        parent_ptr = parent.internalId()
        parent_id = self._int_to_id.get(parent_ptr)
        if not parent_id:
            return False
        
        return self._has_more.get(parent_id, False)
    
    def fetchMore(self, parent: QModelIndex):
        """Fetch more children for virtual scrolling."""
        if not parent.isValid():
            return
        
        parent_ptr = parent.internalId()
        parent_id = self._int_to_id.get(parent_ptr)
        if not parent_id:
            return
        
        # Trigger async fetch
        asyncio.ensure_future(self._fetch_more_children(parent_id))
    
    async def _fetch_more_children(self, parent_id: str):
        """Fetch next batch of children."""
        try:
            current_count = len(self._children.get(parent_id, []))
            
            # Get next batch
            more_children = await self.fs_service.get_children(
                ObjectId(parent_id), 
                limit=self.FETCH_SIZE,
                skip=current_count
            )
            
            if not more_children:
                self._has_more[parent_id] = False
                return
            
            # Check if there's still more
            self._has_more[parent_id] = len(more_children) == self.FETCH_SIZE
            
            # Get parent index
            parent_index = self._find_index(parent_id)
            if not parent_index.isValid():
                # Just extend without notification
                self._children[parent_id].extend(more_children)
                for child in more_children:
                    self._cache[str(child._id)] = child
                self._pending_updates = True
                return
            
            # Insert new rows
            start_row = current_count
            end_row = start_row + len(more_children) - 1
            
            self.beginInsertRows(parent_index, start_row, end_row)
            self._children[parent_id].extend(more_children)
            for child in more_children:
                self._cache[str(child._id)] = child
            self.endInsertRows()
            
            logger.debug(f"Fetched {len(more_children)} more items for {parent_id}")
            
        except Exception as e:
            logger.exception(f"Error fetching more children: {e}")
    
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """Get data for index."""
        try:
            if not index.isValid():
                return None
            
            int_ptr = index.internalId()
            record_id_str = self._int_to_id.get(int_ptr)
            if not record_id_str:
                return None
            
            record = self._cache.get(record_id_str)
            
            if not record:
                return None
            
            column = index.column()
            
            if role == Qt.DisplayRole:
                if column == 0:
                    return record.name or "Unknown"
                elif column == 1:
                    size = getattr(record, 'size_bytes', 0)
                    return self._format_size(size if size else 0)
                elif column == 2:
                    if hasattr(record, 'is_library_root'):
                        return "Library Root"
                    return getattr(record, 'driver_type', "Unknown")
                elif column == 3:
                    mod_time = getattr(record, 'modified_at', None)
                    return mod_time.strftime("%Y-%m-%d %H:%M") if mod_time else ""
            
            elif role == Qt.DecorationRole:
                if column == 0:
                    return self._get_icon(record)
            
            elif role == self.PathRole:
                return getattr(record, 'path', None)
            
            elif role == self.IdRole:
                return record_id_str
            
            elif role == self.IsDirectoryRole:
                return hasattr(record, 'is_library_root') or hasattr(record, 'child_count')
            
            elif role == self.SizeRole:
                return getattr(record, 'size_bytes', 0)
            
            return None
            
        except Exception as e:
            logger.exception(f"Error in data() for index {index.row()},{index.column()}: {e}")
            return None
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        """Get header data."""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            headers = ["Name", "Size", "Type", "Modified"]
            if section < len(headers):
                return headers[section]
        return None
    
    def canFetchMore(self, parent: QModelIndex) -> bool:
        """Check if more data can be fetched."""
        if not parent.isValid():
            return False
        
        parent_ptr = parent.internalId()
        parent_id_str = self._int_to_id.get(parent_ptr)
        return parent_id_str not in self._children
    
    def fetchMore(self, parent: QModelIndex):
        """Fetch more data."""
        if not parent.isValid():
            return
        
        parent_ptr = parent.internalId()
        parent_id_str = self._int_to_id.get(parent_ptr)
        if parent_id_str:
            asyncio.ensure_future(self._load_children(parent_id_str))
    
    async def _load_children(self, parent_id: str):
        """Load children for parent."""
        if parent_id in self._children:
            return
        
        try:
            # Get parent record
            parent_record = self._cache.get(parent_id)
            if not parent_record:
                logger.warning(f"Parent record not found: {parent_id}")
                self._children[parent_id] = []
                return
            
            # Load children from FSService with virtualization
            # Only load first batch, more loaded via fetchMore()
            children = await self.fs_service.get_children(
                ObjectId(parent_id), 
                limit=self.FETCH_SIZE
            )
            
            if not children:
                self._children[parent_id] = []
                self._has_more[parent_id] = False
                return
            
            # Track if more items available (for virtualization)
            self._has_more[parent_id] = len(children) == self.FETCH_SIZE
            if self._has_more[parent_id]:
                logger.info(f"Folder has {self.FETCH_SIZE}+ items, virtualization enabled")

            # Find parent index for insertion
            parent_index = self._find_index(parent_id)
            if not parent_index.isValid() and parent_id not in [str(r._id) for r in self._roots]:
                logger.warning(f"Parent index invalid for {parent_id} during load")
                # Fallback: cache children but DON'T emit layoutChanged (causes freeze)
                # Instead, mark as pending update
                self._children[parent_id] = children
                for child in children:
                    self._cache[str(child._id)] = child
                # Schedule deferred update instead of immediate emit
                self._pending_updates = True
                if not self._update_timer.isActive():
                    self._update_timer.start()
                return

            # Begin insertion
            self.beginInsertRows(parent_index, 0, len(children) - 1)
            
            # Cache children
            self._children[parent_id] = children
            for child in children:
                self._cache[str(child._id)] = child
            
            self.endInsertRows()
            
            logger.debug(f"Loaded {len(children)} children for {parent_record.name}")
                
        except Exception as e:
            logger.exception(f"Failed to load children for {parent_id}: {e}")
            self._children[parent_id] = []  # Set empty to prevent retries
    
    def _find_index(self, record_id: str) -> QModelIndex:
        """Find index for record ID."""
        # Check roots
        for i, root in enumerate(self._roots):
            if str(root._id) == record_id:
                int_id = self._register_id(record_id)
                return self.createIndex(i, 0, int_id)
        
        # Check children
        for parent_id, children in self._children.items():
            for i, child in enumerate(children):
                if str(child._id) == record_id:
                    # Recursion might be expensive but okay for finding parent
                    parent_index = self._find_index(parent_id)
                    return self.index(i, 0, parent_index)
        
        return QModelIndex()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def set_drag_state(self, active: bool):
        """
        Set drag state to pause/resume thumbnail loading.
        
        Args:
            active: True when drag starts, False when drag ends
        """
        self._drag_in_progress = active
        if not active:
            # Reason: Resume any pending thumbnail loads after drag ends
            logger.debug("Drag ended, thumbnail loading resumed")
    
    def _get_icon(self, record) -> QIcon:
        """Get icon for record (synchronous/cached part)."""
        # 1. Directory icon
        if hasattr(record, 'is_library_root') or hasattr(record, 'child_count'):
            return QIcon.fromTheme("folder")
        
        # 2. Check cache for thumbnail
        record_id = str(record._id)
        if record_id in self._thumbnail_cache:
            return self._thumbnail_cache[record_id]
        
        # 3. Skip async load during drag to prevent UI freeze
        if self._drag_in_progress:
            return QIcon.fromTheme("text-x-generic")
        
        # 4. Trigger async load if not already loading
        if record_id not in self._loading_thumbnails:
            self._loading_thumbnails.add(record_id)
            asyncio.ensure_future(self._load_thumbnail(record_id))
            
        # 5. Return default file icon
        return QIcon.fromTheme("text-x-generic")

    async def _load_thumbnail(self, record_id: str):
        """Async load thumbnail."""
        try:
            # Check if likely image
            record = self._cache.get(record_id)
            if not record: 
                return
                
            ext = record.extension.lower() if hasattr(record, 'extension') else ""
            if ext not in ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif']:
                 return
            
            # Request from service
            thumb_bytes = await self.thumbnail_service.get_or_create(
                record._id, 
                size=256
            )
            
            if thumb_bytes:
                # Create QIcon from bytes
                pixmap = QPixmap()
                pixmap.loadFromData(thumb_bytes)
                icon = QIcon(pixmap)
                
                # Update cache
                self._thumbnail_cache[record_id] = icon
                
                # Notify view (find index and emit dataChanged)
                # Skip notification during drag to prevent UI freeze
                if self._drag_in_progress:
                    return
                
                index = self._find_index(record_id)
                if index.isValid():
                    self.dataChanged.emit(index, index, [Qt.DecorationRole])
                    
        except Exception as e:
            logger.warning(f"Failed to load thumbnail for {record_id}: {e}")
        finally:
            self._loading_thumbnails.discard(record_id)
    
    # ==================== Drag and Drop Support ====================
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Return item flags for index."""
        default_flags = super().flags(index)
        
        if index.isValid():
            # Enable dragging for all items
            return default_flags | Qt.ItemIsDragEnabled
        
        return default_flags
    
    def mimeTypes(self) -> list:
        """Return supported MIME types for drag."""
        return ['application/x-file-ids', 'text/uri-list']
    
    def mimeData(self, indexes: list):
        """Create MIME data for dragged items."""
        from PySide6.QtCore import QMimeData, QByteArray, QUrl
        
        mime_data = QMimeData()
        file_ids = []
        urls = []
        
        for index in indexes:
            if index.isValid() and index.column() == 0:  # Only first column
                int_ptr = index.internalId()
                record_id = self._int_to_id.get(int_ptr)
                if record_id:
                    file_ids.append(record_id)
                    
                    # Also get path for uri-list
                    record = self._cache.get(record_id)
                    if record and hasattr(record, 'path'):
                        urls.append(QUrl.fromLocalFile(record.path))
        
        if file_ids:
            # Custom format for internal use
            data = ','.join(file_ids).encode('utf-8')
            mime_data.setData('application/x-file-ids', QByteArray(data))
            
            # Standard URL format for external apps
            if urls:
                mime_data.setUrls(urls)
        
        return mime_data

