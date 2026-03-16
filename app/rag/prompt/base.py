from __future__ import annotations

from abc import ABC, abstractmethod

from app.llm.schemas import LLMMessage
from app.rag.schemas import RetrievedChunk


class BasePrompt(ABC):
    prompt_name: str

    @abstractmethod
    def build_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[LLMMessage]:
        raise NotImplementedError
