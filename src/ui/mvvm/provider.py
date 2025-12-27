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
        if vm_cls not in self._cache:
            self._cache[vm_cls] = vm_cls(self.locator)
        return self._cache[vm_cls]
