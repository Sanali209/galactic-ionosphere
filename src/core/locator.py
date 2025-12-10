from .events import ObserverEvent
from .messaging.builder import MessageBuilder, SystemMessage
from .config import ConfigManager
from .capabilities.base import CoreFacade

class ServiceLocator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceLocator, cls).__new__(cls)
            cls._instance.is_ready = False
        return cls._instance

    def init(self, config_path: str):
        if self.is_ready: return
        
        # 1. Global Message Bus
        self.bus = ObserverEvent("SystemBus")
        
        # 2. Tools
        self.msg_builder = MessageBuilder()
        self.config = ConfigManager(config_path)
        
        # 3. Capability Facade
        self.caps = CoreFacade()
        
        # 4. Reactive binding: Config -> Facade
        # If config changes, switching drivers etc
        self.config.on_changed.connect(self._on_config_change)
        
        self.is_ready = True

    def broadcast(self, msg: SystemMessage):
        """Send message to all subscribers (UI, Logs)"""
        self.bus.emit(msg)

    def _on_config_change(self, section, key, value):
        # Auto-switch logic
        if section == "ai" and key == "provider_id":
            # In a real scenario we'd have error handling here
            try:
                self.caps.ai_vectors.switch(value)
            except Exception as e:
                print(f"Failed to switch driver: {e}")

# Global access
sl = ServiceLocator()
