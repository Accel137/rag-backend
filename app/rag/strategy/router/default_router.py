from __future__ import annotations

from app.rag.schemas import RAGStrategy
from app.rag.strategy.router.base import BaseRouter


class DefaultRouter(BaseRouter):
    def route(self, query: str) -> RAGStrategy:
        return RAGStrategy(
            retriever_name="vector",
            prompt_name="default",
        )
