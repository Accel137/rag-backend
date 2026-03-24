from __future__ import annotations

import math
from dataclasses import dataclass

from app.rag.schemas import DocumentChunk, RetrievedChunk
from app.rag.vector_store.base import BaseVectorStore


@dataclass
class StoredVectorRecord:
    chunk: DocumentChunk
    vector: list[float]


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._records: list[StoredVectorRecord] = []

    async def add(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for chunk, vector in zip(chunks, vectors):
            self._records.append(StoredVectorRecord(chunk=chunk, vector=vector))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        matched_records = [
            record for record in self._records if self._matches_filters(record, filters)
        ]

        scored_records = sorted(
            (
                (
                    self._cosine_similarity(query_vector, record.vector),
                    record,
                )
                for record in matched_records
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            RetrievedChunk(
                id=record.chunk.id,
                text=record.chunk.text,
                score=score,
                metadata=record.chunk.metadata.model_dump(mode="json"),
            )
            for score, record in scored_records[:top_k]
        ]

    async def delete_by_document_id(self, document_id: str) -> int:
        original_count = len(self._records)
        self._records = [
            record
            for record in self._records
            if record.chunk.metadata.document_id != document_id
        ]
        return original_count - len(self._records)

    @staticmethod
    def _matches_filters(
        record: StoredVectorRecord,
        filters: dict[str, str] | None,
    ) -> bool:
        if not filters:
            return True

        metadata = record.chunk.metadata.model_dump(mode="json")
        for key, value in filters.items():
            candidate = metadata.get(key)
            if isinstance(candidate, list):
                if value not in candidate:
                    return False
                continue

            if candidate is None or str(candidate) != value:
                return False

        return True

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2):
            raise ValueError("query vector and stored vector must have the same length")

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

