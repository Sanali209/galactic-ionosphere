from abc import ABC, abstractmethod
from typing import TypeVar, Generic

class ICommand(ABC):
    pass

C = TypeVar('C', bound=ICommand)

class ICommandHandler(Generic[C], ABC):
    @abstractmethod
    async def handle(self, command: C):
        pass
