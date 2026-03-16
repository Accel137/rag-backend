from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.schemas import RAGStrategy


class BaseRouter(ABC):
    @abstractmethod
    def route(self, query: str) -> RAGStrategy:
        raise NotImplementedError
