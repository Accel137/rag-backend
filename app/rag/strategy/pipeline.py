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

    async def decide(self, messages: list[ChatMessage]) -> RAGPlan:
        query = await self._rewriter.rewrite(messages) or ""
        return RAGPlan(
            query=query,
            strategy=await self._router.route(query),
        )
