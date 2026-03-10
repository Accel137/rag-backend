from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.rag.schemas import DocumentMetadata, DocumentChunk


class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str, metadata: DocumentMetadata) -> list[DocumentChunk]:
        raise NotImplementedError
