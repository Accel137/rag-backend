from __future__ import annotations

from app.rag.strategy.rewriter import BaseRewriter, DefaultRewriter
from app.rag.strategy.router import BaseRouter, DefaultRouter
from app.rag.strategy.schemas import RAGPlan
from app.schemas.chat import ChatMessage


class StrategyPipeline:
    def __init__(
        self,
        rewriter: BaseRewriter | None = None,
        router: BaseRouter | None = None,
    ) -> None:
        self._rewriter = rewriter or DefaultRewriter()
        self._router = router or DefaultRouter()

    def decide(self, messages: list[ChatMessage]) -> RAGPlan:
        query = self._rewriter.rewrite(messages) or ""
        return RAGPlan(
            query=query,
            strategy=self._router.route(query),
        )
