from app.rag.embedding import MockEmbedder
from app.rag.retriever import Retriever
from app.rag.schemas import ChunkMetadata, DocumentChunk, RAGQuery
from app.rag.vector_store import InMemoryVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    source: str = "unit-test",
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            source=source,
            chunk_index=chunk_index,
        ),
    )


def test_retrieve_returns_ranked_chunks():
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    chunks = [
        make_chunk("chunk-1", "apple fruit"),
        make_chunk("chunk-2", "car engine and vehicle repair"),
        make_chunk("chunk-3", "apple orchard and fresh fruit"),
    ]
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store.add(chunks, vectors)

    results = retriever.retrieve("apple fruit", top_k=2)

    assert len(results) == 2
    assert results[0].id == "chunk-1"
    assert results[0].score >= results[1].score


def test_retrieve_returns_empty_list_for_blank_query():
    embedder = MockEmbedder()
    retriever = Retriever(embedder=embedder, vector_store=InMemoryVectorStore())

    assert retriever.retrieve("   ") == []


def test_retrieve_passes_filters_to_vector_store():
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    chunks = [
        make_chunk("chunk-1", "apple fruit and nutrition", document_id="doc-1"),
        make_chunk("chunk-2", "apple orchard and harvest", document_id="doc-2"),
    ]
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store.add(chunks, vectors)

    results = retriever.retrieve(
        "apple",
        top_k=5,
        filters={"document_id": "doc-2"},
    )

    assert len(results) == 1
    assert results[0].id == "chunk-2"


def test_retrieve_from_query_uses_rag_query_top_k():
    embedder = MockEmbedder()
    vector_store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    chunks = [
        make_chunk("chunk-1", "apple fruit and nutrition"),
        make_chunk("chunk-2", "apple orchard and harvest"),
        make_chunk("chunk-3", "apple pie recipe"),
    ]
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store.add(chunks, vectors)

    results = retriever.retrieve_from_query(RAGQuery(query="apple", top_k=2))

    assert len(results) == 2
