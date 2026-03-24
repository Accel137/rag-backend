from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.schemas import DocumentChunk, RetrievedChunk


class BaseVectorStore(ABC):
    @abstractmethod
    async def add(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> int:
        raise NotImplementedError


