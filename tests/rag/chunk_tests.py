from app.rag.schemas import DocumentChunk


def get_chunk_text(chunk: DocumentChunk) -> str:
    """
    兼容不同字段命名：
    - chunk.text
    - chunk.content
    """
    if hasattr(chunk, "text"):
        return chunk.text
    if hasattr(chunk, "content"):
        return chunk.content
    raise AssertionError("DocumentChunk must have either 'text' or 'content' field")


def test_split_short_text_returns_single_chunk(chunker, sample_metadata):
    text = "hello world"

    chunks = chunker.split(text, sample_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert get_chunk_text(chunks[0]) == text


def test_split_empty_text_returns_empty_list(chunker, sample_metadata):
    chunks = chunker.split("", sample_metadata)

    assert isinstance(chunks, list)
    assert chunks == []


def test_split_whitespace_text(chunker, sample_metadata):
    text = "   \n\t   "

    chunks = chunker.split(text, sample_metadata)

    assert isinstance(chunks, list)
    # 接受两种实现：
    # 1. 返回 []
    # 2. 返回一个保留原始空白的 chunk
    assert chunks == [] or (
        len(chunks) == 1 and get_chunk_text(chunks[0]) == text
    )


def test_split_long_text_into_multiple_chunks(chunker, sample_metadata):
    text = "A" * 2000

    chunks = chunker.split(text, sample_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(len(get_chunk_text(c)) > 0 for c in chunks)


def test_split_chinese_text(chunker, sample_metadata):
    text = "这是一个用于测试分块器的中文文本。" * 200

    chunks = chunker.split(text, sample_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(get_chunk_text(c)) > 0 for c in chunks)


def test_split_mixed_language_text(chunker, sample_metadata):
    text = (
        "这是中文。This is English. 这是一个 mixed language 的 chunking test. "
        "RAG 系统通常需要同时处理中文和英文。"
    ) * 50

    chunks = chunker.split(text, sample_metadata)

    assert isinstance(chunks, list)
    assert len(chunks) > 1

    merged = "".join(get_chunk_text(c) for c in chunks)
    assert "中文" in merged
    assert "English" in merged


def test_split_preserves_order(chunker, sample_metadata):
    text = "part1-" + ("A" * 600) + "-part2-" + ("B" * 600) + "-part3"

    chunks = chunker.split(text, sample_metadata)
    merged = "".join(get_chunk_text(c) for c in chunks)

    assert "part1" in merged
    assert "part2" in merged
    assert "part3" in merged
    assert merged.find("part1") < merged.find("part2") < merged.find("part3")


def test_split_no_empty_chunks(chunker, sample_metadata):
    text = ("abc " * 1000).strip()

    chunks = chunker.split(text, sample_metadata)

    assert len(chunks) > 0
    assert all(get_chunk_text(c) != "" for c in chunks)


def test_split_overlap_behavior_if_enabled(chunker, sample_metadata):
    text = "0123456789" * 200

    chunks = chunker.split(text, sample_metadata)

    if len(chunks) >= 2:
        first = get_chunk_text(chunks[0])
        second = get_chunk_text(chunks[1])

        # 不强绑具体 overlap 值，只检查有一定重叠迹象
        assert first[-20:] in second or first[-10:] in second


def test_split_returns_metadata_related_fields_if_present(chunker, sample_metadata):
    text = "hello world " * 200

    chunks = chunker.split(text, sample_metadata)

    assert len(chunks) > 0

    # 这里不强制字段一定存在，但如果你有这些字段就顺便测一下
    first = chunks[0]

    if hasattr(first, "metadata"):
        assert first.metadata is not None

    if hasattr(first, "chunk_index"):
        assert first.chunk_index == 0


def test_split_single_character_text(chunker, sample_metadata):
    chunks = chunker.split("a", sample_metadata)

    assert len(chunks) == 1
    assert get_chunk_text(chunks[0]) == "a"