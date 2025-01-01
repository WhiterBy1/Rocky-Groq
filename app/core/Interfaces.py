from abc import ABC, abstractmethod

class Service(ABC):
    @abstractmethod
    def start(self):
        pass
    @abstractmethod
    def configure(self):al
        pass
class Utils(ABC):
    @abstractmethod
    def load_config(self):
        pass