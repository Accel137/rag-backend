from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.schemas import RetrievedChunk


class BaseRetriever(ABC):
    retriever_name: str

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
