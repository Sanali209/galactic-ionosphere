from PySide6.QtCore import QObject, Signal

class BaseViewModel(QObject):
    """
    Base class for ViewModels.
    Provides infrastructure for property change notifications and
    binding to the Locator.
    """
    def __init__(self, locator):
        super().__init__()
        self.locator = locator
    
    def on_property_changed(self, property_name: str, value):
        """
        Helper to emit signals dynamically if needed, 
        though explicit signals are preferred in Qt.
        """
        pass
