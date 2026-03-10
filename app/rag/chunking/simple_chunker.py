from __future__ import annotations

import uuid
from typing import Any

from app.rag.chunking.base import BaseChunker
from app.rag.schemas import DocumentChunk, DocumentMetadata, ChunkMetadata


class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: DocumentMetadata) -> list[DocumentChunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = ChunkMetadata(
                    **metadata.model_dump(),
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )

                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

            start += step

        return chunks