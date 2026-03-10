import math


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def test_embed_texts_empty_input_returns_empty_list(embedder):
    vectors = embedder.embed_texts([])

    assert isinstance(vectors, list)
    assert vectors == []


def test_embed_texts_single_text(embedder):
    texts = ["hello world"]

    vectors = embedder.embed_texts(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert len(vectors[0]) == embedder.dimension
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_texts_multiple_texts(embedder):
    texts = [
        "hello world",
        "rag system",
        "embedding test",
    ]

    vectors = embedder.embed_texts(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) for v in vectors)
    assert all(len(v) == embedder.dimension for v in vectors)


def test_embed_texts_output_dimension_matches_property(embedder):
    vectors = embedder.embed_texts(["hello", "world"])

    assert len(vectors) == 2
    assert all(len(v) == embedder.dimension for v in vectors)


def test_embed_texts_same_input_is_deterministic(embedder):
    v1 = embedder.embed_texts(["hello world"])[0]
    v2 = embedder.embed_texts(["hello world"])[0]

    assert v1 == v2


def test_embed_texts_different_inputs_not_identical(embedder):
    v1 = embedder.embed_texts(["apple"])[0]
    v2 = embedder.embed_texts(["car engine"])[0]

    assert v1 != v2


def test_embed_texts_chinese_text(embedder):
    vectors = embedder.embed_texts(["这是一个中文测试文本。"])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimension


def test_embed_texts_mixed_language_text(embedder):
    vectors = embedder.embed_texts(["这是中文 and this is English together."])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimension


def test_embed_texts_output_contains_only_floats(embedder):
    vectors = embedder.embed_texts(["hello world", "another sentence"])

    for vector in vectors:
        assert all(isinstance(x, float) for x in vector)
        assert all(not math.isnan(x) for x in vector)
        assert all(not math.isinf(x) for x in vector)


def test_embed_texts_preserves_input_order(embedder):
    texts = ["first text", "second text", "third text"]

    vectors = embedder.embed_texts(texts)

    single_1 = embedder.embed_texts(["first text"])[0]
    single_2 = embedder.embed_texts(["second text"])[0]
    single_3 = embedder.embed_texts(["third text"])[0]

    assert vectors[0] == single_1
    assert vectors[1] == single_2
    assert vectors[2] == single_3


def test_embed_query_returns_single_vector(embedder):
    vector = embedder.embed_query("hello world")

    assert isinstance(vector, list)
    assert len(vector) == embedder.dimension
    assert all(isinstance(x, float) for x in vector)


def test_embed_query_is_deterministic(embedder):
    v1 = embedder.embed_query("what is rag")
    v2 = embedder.embed_query("what is rag")

    assert v1 == v2


def test_embed_query_and_text_have_same_dimension(embedder):
    text_vector = embedder.embed_texts(["what is rag"])[0]
    query_vector = embedder.embed_query("what is rag")

    assert len(text_vector) == embedder.dimension
    assert len(query_vector) == embedder.dimension


def test_embed_query_output_contains_only_floats(embedder):
    vector = embedder.embed_query("test query")

    assert all(isinstance(x, float) for x in vector)
    assert all(not math.isnan(x) for x in vector)
    assert all(not math.isinf(x) for x in vector)


def test_embed_query_semantic_similarity_basic(embedder):
    v1 = embedder.embed_query("apple fruit")
    v2 = embedder.embed_query("apple")
    v3 = embedder.embed_query("car engine")

    sim12 = cosine_similarity(v1, v2)
    sim13 = cosine_similarity(v1, v3)

    # 只有真正语义 embedding 才一定成立
    # 如果你现在是 mock/hash embedder，这条可能失败
    assert sim12 > sim13