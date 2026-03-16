import pytest

from app.rag.schemas import ChunkMetadata, DocumentChunk
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


def test_build_vector_store_returns_in_memory_store(vector_store):
    assert isinstance(vector_store, InMemoryVectorStore)


def test_add_requires_matching_chunk_and_vector_lengths(vector_store):
    chunk = make_chunk("chunk-1", "hello")

    with pytest.raises(ValueError, match="same length"):
        vector_store.add([chunk], [])


def test_add_and_search_returns_best_matches_in_score_order(vector_store):
    chunks = [
        make_chunk("chunk-1", "alpha", chunk_index=0),
        make_chunk("chunk-2", "beta", chunk_index=1),
        make_chunk("chunk-3", "gamma", chunk_index=2),
    ]
    vectors = [
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
    ]

    vector_store.add(chunks, vectors)

    results = vector_store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].id == "chunk-1"
    assert results[1].id == "chunk-2"
    assert results[0].score >= results[1].score


def test_search_supports_metadata_filters(vector_store):
    chunks = [
        make_chunk("chunk-1", "alpha", document_id="doc-1", source="kb-a"),
        make_chunk("chunk-2", "beta", document_id="doc-2", source="kb-b"),
    ]
    vectors = [
        [1.0, 0.0],
        [1.0, 0.0],
    ]

    vector_store.add(chunks, vectors)

    results = vector_store.search([1.0, 0.0], filters={"document_id": "doc-2"})

    assert len(results) == 1
    assert results[0].id == "chunk-2"
    assert results[0].metadata["document_id"] == "doc-2"


def test_delete_by_document_id_removes_matching_records(vector_store):
    chunks = [
        make_chunk("chunk-1", "alpha", document_id="doc-1"),
        make_chunk("chunk-2", "beta", document_id="doc-1"),
        make_chunk("chunk-3", "gamma", document_id="doc-2"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    vector_store.add(chunks, vectors)

    deleted_count = vector_store.delete_by_document_id("doc-1")

    assert deleted_count == 2
    remaining = vector_store.search([1.0, 0.0], top_k=10)
    assert [chunk.id for chunk in remaining] == ["chunk-3"]


def test_search_returns_empty_list_for_non_positive_top_k(vector_store):
    chunk = make_chunk("chunk-1", "alpha")
    vector_store.add([chunk], [[1.0, 0.0]])

    assert vector_store.search([1.0, 0.0], top_k=0) == []
