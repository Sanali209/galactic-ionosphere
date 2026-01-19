from typing import Type, TypeVar
from .viewmodel import BaseViewModel

T = TypeVar('T', bound=BaseViewModel)

class ViewModelProvider:
    """
    Factory for creating and managing ViewModel instances,
    ensuring they receive the ServiceLocator.
    """
    def __init__(self, locator):
        self.locator = locator
        self._cache = {}

    def get(self, vm_cls: Type[T]) -> T:
        """
        Get or create ViewModel instance with automatic initialization.
        
        Automatically calls initialize_reactivity() if the ViewModel has this method.
        This ensures ViewModels are properly initialized before use.
        """
        if vm_cls not in self._cache:
            # Create ViewModel instance
            vm = vm_cls(self.locator)
            
            # Automatically call initialize_reactivity() if it exists
            # This fixes the bug where ViewModels weren't being initialized
            if hasattr(vm, 'initialize_reactivity'):
                vm.initialize_reactivity()
            
            self._cache[vm_cls] = vm
        
        return self._cache[vm_cls]
