from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
