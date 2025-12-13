from PySide6.QtCore import QAbstractListModel, Qt, QModelIndex, Signal, Slot, QUrl
import asyncio
from typing import List
from src.core.database.models.image import ImageRecord

class GalleryGridModel(QAbstractListModel):
    """
    Model for the main Image Grid.
    Supports lazy loading (scrolling triggers fetch).
    """
    
    # Roles
    IdRole = Qt.UserRole + 1
    PathRole = Qt.UserRole + 2
    ThumbnailRole = Qt.UserRole + 3
    
    countChanged = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: List[ImageRecord] = []
        self._has_more = True
        self._loading = False
    
    def set_images(self, images: List[ImageRecord]):
        self.beginResetModel()
        self._images = images
        self.endResetModel()
        self.countChanged.emit()

    def add_images(self, new_images: List[ImageRecord]):
        if not new_images:
            return
        
        start = len(self._images)
        end = start + len(new_images) - 1
        
        self.beginInsertRows(QModelIndex(), start, end)
        self._images.extend(new_images)
        self.endInsertRows()
        self.countChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self._images)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        row = index.row()
        if row >= len(self._images):
            return None
            
        img = self._images[row]
        
        if role == self.IdRole:
            return str(img.id)
        elif role == self.PathRole:
            # QML expects 'file:///' prefix for local Image types sometimes, 
            # or just path if using ImageProvider.
            # Let's return clean path.
            return QUrl.fromLocalFile(img.full_path).toString()
        elif role == self.ThumbnailRole:
            # Return hash to build thumbnail path in QML or via Provider
            return img.content_md5
            
        return None

    def roleNames(self):
        return {
            self.IdRole: b"imageId",
            self.PathRole: b"imagePath",
            self.ThumbnailRole: b"thumbnailHash"
        }
