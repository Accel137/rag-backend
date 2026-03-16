from __future__ import annotations

from app.rag.chunking import BaseChunker
from app.rag.embedding import BaseEmbedder
from app.rag.schemas import DocumentChunk, DocumentMetadata, IngestRequest
from app.rag.vector_store import BaseVectorStore


class Ingestor:
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, request: IngestRequest) -> list[DocumentChunk]:
        metadata_payload = dict(request.metadata)
        metadata_payload.pop("document_id", None)
        metadata = DocumentMetadata(
            document_id=request.document_id,
            **metadata_payload,
        )

        chunks = self.chunker.split(request.text, metadata)
        if not chunks:
            return []

        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, vectors)
        return chunks
