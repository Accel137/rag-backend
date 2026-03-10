import pytest

from app.core.config import settings
from app.rag.factory import build_chunker, build_embedder
from app.rag.schemas import DocumentMetadata


@pytest.fixture
def chunker():
    return build_chunker(settings.rag)


@pytest.fixture
def embedder():
    return build_embedder(settings.rag)

@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        source="unit-test",
        document_id="doc-1",
        title="test document",
    )