import pytest
from pydantic import ValidationError

from app.rag.chunking import SimpleChunker
from app.rag.embedding import MockEmbedder
from app.rag.ingest import Ingestor
from app.rag.schemas import IngestRequest
from app.rag.vector_store import InMemoryVectorStore


def test_ingest_adds_chunk_embeddings_to_vector_store():
    chunker = SimpleChunker(chunk_size=20, chunk_overlap=5)
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    ingestor = Ingestor(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    chunks = ingestor.ingest(
        IngestRequest(
            document_id="doc-1",
            text="apple fruit " * 20,
            metadata={"source": "unit-test", "title": "Fruit Notes"},
        )
    )

    assert len(chunks) > 1

    results = vector_store.search(embedder.embed_query("apple fruit"), top_k=3)
    assert len(results) > 0
    assert all(result.metadata["document_id"] == "doc-1" for result in results)


def test_ingest_returns_empty_list_for_blank_text():
    chunker = SimpleChunker(chunk_size=20, chunk_overlap=5)
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    ingestor = Ingestor(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    chunks = ingestor.ingest(
        IngestRequest(
            document_id="doc-1",
            text="   \n\t   ",
            metadata={"source": "unit-test"},
        )
    )

    assert chunks == []
    assert vector_store.search(embedder.embed_query("apple"), top_k=5) == []


def test_ingest_requires_source_in_metadata():
    chunker = SimpleChunker(chunk_size=20, chunk_overlap=5)
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    ingestor = Ingestor(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(ValidationError):
        ingestor.ingest(
            IngestRequest(
                document_id="doc-1",
                text="apple fruit",
                metadata={},
            )
        )
