from PySide6.QtCore import QAbstractListModel, Qt, Slot, Signal, QModelIndex
from datetime import datetime
from src.core.locator import sl

class JournalViewModel(QAbstractListModel):
    LogsChanged = Signal()
    
    LevelRole = Qt.UserRole + 1
    CategoryRole = Qt.UserRole + 2
    MessageRole = Qt.UserRole + 3
    TimestampRole = Qt.UserRole + 4
    DetailsRole = Qt.UserRole + 5

    def __init__(self):
        super().__init__()
        self._all_logs = [] # Source of truth
        self._filtered_logs = [] # View
        self._filter_text = ""
        self._active_levels = {"INFO", "WARNING", "ERROR", "SUCCESS"}
        
        # Connect to service if available
        # Ideally we listen to an event bus or use a specialized sink.
        # For this prototype, we'll poll or use the explicit event bus.
        if sl.bus:
            # We need to bridge 'journal' sink to here. 
            # OR we make JournalViewModel a Sink itself?
            pass

    def add_log(self, record):
        """Called by JournalService or Bus"""
        self.beginInsertRows(QModelIndex(), len(self._filtered_logs), len(self._filtered_logs))
        self._all_logs.append(record)
        if self._matches_filter(record):
            self._filtered_logs.append(record)
        self.endInsertRows()
        # Scroll?

    def rowCount(self, parent=QModelIndex()):
        return len(self._filtered_logs)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        rec = self._filtered_logs[index.row()]
        
        if role == Qt.DisplayRole:
            return f"[{rec.timestamp.strftime('%H:%M:%S')}] [{rec.level}] {rec.message}"
        
        if role == self.LevelRole: return rec.level
        if role == self.CategoryRole: return rec.category
        if role == self.MessageRole: return rec.message
        if role == self.TimestampRole: return rec.timestamp.strftime("%H:%M:%S")
        if role == self.DetailsRole: return str(rec.details)
        return None

    def roleNames(self):
        return {
            self.LevelRole: b"level",
            self.CategoryRole: b"category",
            self.MessageRole: b"message",
            self.TimestampRole: b"timestamp",
            self.DetailsRole: b"details"
        }

    def _matches_filter(self, rec):
        if rec.level not in self._active_levels: return False
        if self._filter_text and self._filter_text.lower() not in rec.message.lower(): return False
        return True

    @Slot(str)
    def set_filter(self, text):
        self._filter_text = text
        self._refresh()

    @Slot(str, bool)
    def toggle_level(self, level, active):
        if active: self._active_levels.add(level)
        else: self._active_levels.discard(level)
        self._refresh()
        
    @Slot()
    def clear(self):
        self.beginResetModel()
        self._all_logs.clear()
        self._filtered_logs.clear()
        self.endResetModel()

    def _refresh(self):
        self.beginResetModel()
        self._filtered_logs = [r for r in self._all_logs if self._matches_filter(r)]
        self.endResetModel()
