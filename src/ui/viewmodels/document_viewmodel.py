"""
DocumentViewModel - MVVM pattern for documents

ViewModels handle business logic, DockingService handles UI.
"""
from PySide6.QtCore import QObject, Signal, Property
from typing import Optional


class DocumentViewModel(QObject):
    """
    ViewModel for a document.
    
    Separates document business logic from UI presentation.
    Use with DockingService for MVVM-compliant docking.
    """
    
    title_changed = Signal(str)
    content_changed = Signal()
    dirty_changed = Signal(bool)
    
    def __init__(self, doc_id: str, title: str):
        super().__init__()
        self._doc_id = doc_id
        self._title = title
        self._is_dirty = False
        self._content = ""
    
    @Property(str, notify=title_changed)
    def title(self) -> str:
        return self._title
    
    @title.setter
    def title(self, value: str):
        if self._title != value:
            self._title = value
            self.title_changed.emit(value)
    
    @Property(bool, notify=dirty_changed)
    def is_dirty(self) -> bool:
        return self._is_dirty
    
    @is_dirty.setter
    def is_dirty(self, value: bool):
        if self._is_dirty != value:
            self._is_dirty = value
            self.dirty_changed.emit(value)
    
    @Property(str)
    def doc_id(self) -> str:
        return self._doc_id
