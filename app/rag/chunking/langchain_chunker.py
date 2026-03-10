from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.schemas import DocumentMetadata, ChunkMetadata, DocumentChunk
from app.rag.chunking import BaseChunker


class LangChainChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(
        self,
        text: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:

        text = text.strip()
        if not text:
            return []

        raw_chunks = self.splitter.split_text(text)

        chunks: list[DocumentChunk] = []
        start = 0

        for idx, chunk_text in enumerate(raw_chunks):

            # 找到 chunk 在原文中的位置
            pos = text.find(chunk_text, start)

            if pos == -1:
                pos = start

            start_char = pos
            end_char = pos + len(chunk_text)

            start = end_char

            chunk_metadata = ChunkMetadata(
                **metadata.model_dump(),
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
            )

            chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        return chunks