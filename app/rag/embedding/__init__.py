from app.rag.embedding.base import BaseEmbedder
from app.rag.embedding.mock_embedder import MockEmbedder

__all__ = ["BaseEmbedder", "MockEmbedder"]

try:
    from app.rag.embedding.openai_embedder import OpenAIEmbedder
except ModuleNotFoundError:
    OpenAIEmbedder = None
else:
    __all__.append("OpenAIEmbedder")
