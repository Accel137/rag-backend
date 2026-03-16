from __future__ import annotations

from app.rag.strategy.rewriter.base import BaseRewriter
from app.schemas.chat import ChatMessage


class DefaultRewriter(BaseRewriter):
    def rewrite(self, messages: list[ChatMessage]) -> str | None:
        for message in reversed(messages):
            if message.role == "user":
                content = message.content.strip()
                if content:
                    return content
        return None
