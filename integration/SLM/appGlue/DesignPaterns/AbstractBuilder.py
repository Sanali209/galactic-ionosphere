from abc import ABC, abstractmethod


class Builder(ABC):
    def __init__(self, product=None):
        self._product = product

    def build(self):
        return self._product




