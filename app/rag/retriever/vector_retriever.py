from __future__ import annotations

from app.rag.embedding.base import BaseEmbedder
from app.rag.retriever.base import BaseRetriever
from app.rag.schemas import RetrievedChunk
from app.rag.vector_store.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self.retriever_name = "vector"
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
