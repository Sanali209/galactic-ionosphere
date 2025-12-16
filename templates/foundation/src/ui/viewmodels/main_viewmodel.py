from PySide6.QtCore import Signal, Slot
from loguru import logger
from ..mvvm.viewmodel import BaseViewModel

class MainViewModel(BaseViewModel):
    """
    ViewModel for the MainWindow.
    Manages state for the active view, status messages, and command execution.
    """
    # Signals to notify View of changes
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

    @Slot(str)
    def log_action(self, msg: str):
        """
        Example slot called from UI.
        """
        logger.info(f"ViewModel Log: {msg}")
        self.status_message = f"Last Action: {msg}"
