from app.core.config import RAGSettings
from app.rag.chunking import BaseChunker, SimpleChunker, LangChainChunker
from app.rag.embedding import BaseEmbedder, MockEmbedder, OpenAIEmbedder


def build_chunker(config: RAGSettings) -> BaseChunker:
    provider = config.chunk_provider.lower()

    if provider == "simple":
        return SimpleChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    if provider == "langchain":
        return LangChainChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    raise ValueError(f"Unsupported chunker provider: {config.provider}")


def build_embedder(config: RAGSettings) -> BaseEmbedder:
    provider = config.embedding_provider.lower()

    if provider == "mock":
        return MockEmbedder()
    
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=config.openai_api_key,
            model=config.embedding_model
        )

    raise ValueError(f"Unsupported embedder provider: {config.provider}")