from app.core.config import RAGSettings
from app.rag.chunking.base import BaseChunker
from app.rag.chunking.simple_chunker import SimpleChunker
from app.rag.embedding.base import BaseEmbedder
from app.rag.embedding.mock_embedder import MockEmbedder
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import DefaultPrompt
from app.rag.retriever import VectorRetriever
from app.rag.registry import PromptRegistry, RetrieverRegistry
from app.rag.strategy import StrategyPipeline
from app.rag.vector_store.base import BaseVectorStore
from app.rag.vector_store.memory_store import InMemoryVectorStore
from app.rag.ingest import Ingestor
from dataclasses import dataclass

@dataclass
class RAGModule:
    ingestor: Ingestor
    rag_pipeline: RAGPipeline
    strategy_pipeline: StrategyPipeline



def build_chunker(config: RAGSettings) -> BaseChunker:
    provider = config.chunk_provider.lower()

    if provider == "simple":
        return SimpleChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    if provider == "langchain":
        from app.rag.chunking.langchain_chunker import LangChainChunker

        return LangChainChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    raise ValueError(f"Unsupported chunker provider: {config.chunk_provider}")


def build_embedder(config: RAGSettings) -> BaseEmbedder:
    provider = config.embedding_provider.lower()

    if provider == "mock":
        return MockEmbedder()
    
    if provider == "openai":
        from app.rag.embedding.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(
            api_key=config.openai_api_key,
            model=config.embedding_model
        )

    raise ValueError(f"Unsupported embedder provider: {config.embedding_provider}")


def build_vector_store(config: RAGSettings) -> BaseVectorStore:
    provider = config.vector_store_provider.lower()

    if provider == "memory":
        return InMemoryVectorStore()

    raise ValueError(
        f"Unsupported vector store provider: {config.vector_store_provider}"
    )


def build_retriever_registry(embedder: BaseEmbedder, vector_store: BaseVectorStore) -> RetrieverRegistry:
    registry = RetrieverRegistry()
    registry.register(VectorRetriever(embedder, vector_store))
    return registry


def build_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(DefaultPrompt())
    return registry


def build_rag_pipeline(retriever_registry: RetrieverRegistry, prompt_registry: PromptRegistry) -> RAGPipeline:
    rag_pipeline = RAGPipeline(retriever_registry, prompt_registry)
    return rag_pipeline

def build_ingestor(chunker: BaseChunker, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> Ingestor:
    rag_ingestor = Ingestor(chunker, embedder, vector_store)
    return rag_ingestor


def build_rag_module(config: RAGSettings) -> RAGModule:
    chunker = build_chunker(config)
    embedder = build_embedder(config)
    vector_store = build_vector_store(config)

    retriever_registry = build_retriever_registry(embedder, vector_store)
    prompt_registry = build_prompt_registry()

    rag_pipeline = build_rag_pipeline(retriever_registry, prompt_registry)
    ingestor = build_ingestor(chunker, embedder, vector_store)
    strategy_pipeline = StrategyPipeline()
    return RAGModule(
        ingestor=ingestor,
        rag_pipeline=rag_pipeline,
        strategy_pipeline=strategy_pipeline,
    )
