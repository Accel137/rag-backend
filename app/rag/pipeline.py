
from __future__ import annotations

from app.llm.schemas import LLMMessage
from app.rag.registry import PromptRegistry, RetrieverRegistry
from app.rag.schemas import RAGStrategy, RetrievedChunk

class RAGPipeline:
    def __init__(
        self,
        retriever_registry: RetrieverRegistry,
        prompt_registry: PromptRegistry,
    ) -> None:
        self.retriever_registry = retriever_registry
        self.prompt_registry = prompt_registry

    

    def retrieve(
        self,
        query: str,
        strategy: RAGStrategy,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        retriever = self.retriever_registry.get(strategy.retriever_name)

        chunks = retriever.retrieve(
            query=query,
            top_k=strategy.top_k,
            filters=filters,
        )
        return chunks

    def build_messages(
        self,
        query: str,
        strategy: RAGStrategy,
        chunks: list[RetrievedChunk],
    ) -> list[LLMMessage]:
        prompt = self.prompt_registry.get(strategy.prompt_name)
        return prompt.build_messages(
            question=query,
            chunks=chunks,
        )
