from app.rag.prompt_builder import DEFAULT_SYSTEM_PROMPT, PromptBuilder
from app.rag.schemas import RetrievedChunk


def make_chunk(
    chunk_id: str,
    text: str,
    score: float = 0.9,
    document_id: str = "doc-1",
    source: str = "unit-test",
    chunk_index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        score=score,
        metadata={
            "document_id": document_id,
            "source": source,
            "chunk_index": chunk_index,
        },
    )


def test_build_context_formats_chunks_with_metadata():
    builder = PromptBuilder()
    chunks = [
        make_chunk("chunk-1", "First chunk text", score=0.95, chunk_index=0),
        make_chunk("chunk-2", "Second chunk text", score=0.85, chunk_index=1),
    ]

    context = builder.build_context(chunks)

    assert "[1]" in context
    assert "document_id=doc-1" in context
    assert "source=unit-test" in context
    assert "chunk_index=1" in context
    assert "First chunk text" in context
    assert "Second chunk text" in context


def test_build_context_returns_fallback_when_no_chunks():
    builder = PromptBuilder()

    assert builder.build_context([]) == "No retrieved context."


def test_build_messages_returns_system_and_user_messages():
    builder = PromptBuilder()
    chunks = [make_chunk("chunk-1", "RAG chunk text")]

    messages = builder.build_messages("What is this about?", chunks)

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content[0].text == DEFAULT_SYSTEM_PROMPT
    assert messages[1].role == "user"
    assert "What is this about?" in messages[1].content[0].text
    assert "RAG chunk text" in messages[1].content[0].text
