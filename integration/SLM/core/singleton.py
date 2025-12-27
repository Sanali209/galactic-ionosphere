"""
Singleton Metaclass Implementation
Thread-safe singleton pattern for core managers
"""

import threading
from typing import Dict, Type, TypeVar, Optional

T = TypeVar('T')


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass
    Ensures only one instance of a class exists
    """
    
    _instances: Dict[Type, object] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Create or return existing singleton instance
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Singleton instance
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def reset(mcs, cls: Type[T]) -> None:
        """
        Reset singleton instance (useful for testing)
        
        Args:
            cls: Class to reset
        """
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]
    
    @classmethod
    def reset_all(mcs) -> None:
        """
        Reset all singleton instances (useful for testing)
        """
        with mcs._lock:
            mcs._instances.clear()
    
    @classmethod
    def get_instance(mcs, cls: Type[T]) -> Optional[T]:
        """
        Get existing singleton instance without creating
        
        Args:
            cls: Class type
            
        Returns:
            Instance if exists, None otherwise
        """
        instance = mcs._instances.get(cls)
        return instance  # type: ignore
    
    @classmethod
    def has_instance(mcs, cls: Type[T]) -> bool:
        """
        Check if singleton instance exists
        
        Args:
            cls: Class type
            
        Returns:
            True if instance exists
        """
        return cls in mcs._instances


class Singleton(metaclass=SingletonMeta):
    """
    Base class for singletons
    Inherit from this to make a class a singleton
    """
    
    @classmethod
    def instance(cls: Type[T]) -> T:
        """
        Get singleton instance (creates if needed)
        
        Returns:
            Singleton instance
        """
        return cls()
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset this singleton instance
        """
        SingletonMeta.reset(cls)
    
    @classmethod
    def exists(cls) -> bool:
        """
        Check if singleton instance exists
        
        Returns:
            True if instance exists
        """
        return SingletonMeta.has_instance(cls)
