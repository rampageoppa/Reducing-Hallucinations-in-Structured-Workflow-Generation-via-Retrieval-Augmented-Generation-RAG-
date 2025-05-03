from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    @abstractmethod
    def build_index(self, candidates: List[str]): ...
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[str]: ...
