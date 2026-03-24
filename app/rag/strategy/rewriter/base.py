from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.chat import ChatMessage


class BaseRewriter(ABC):
    @abstractmethod
    async def rewrite(self, messages: list[ChatMessage]) -> str | None:
        raise NotImplementedError
