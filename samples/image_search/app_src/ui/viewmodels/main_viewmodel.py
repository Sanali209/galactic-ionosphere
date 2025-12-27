"""
Main ViewModel for Image Search application.
"""
from PySide6.QtCore import Signal
from loguru import logger

from src.ui.mvvm.viewmodel import BaseViewModel

class MainViewModel(BaseViewModel):
    """
    ViewModel for Image Search MainWindow.
    """
    statusMessageChanged = Signal(str)
    
    def __init__(self, locator):
        super().__init__(locator)
        self._status_message = "Ready"
    
    @property
    def status_message(self):
        return self._status_message
    
    @status_message.setter
    def status_message(self, value):
        if self._status_message != value:
            self._status_message = value
            self.statusMessageChanged.emit(value)
