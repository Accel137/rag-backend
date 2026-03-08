# app/llm/providers/base.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.llm.schemas import LLMRequest, LLMResponse, StreamEvent


class BaseLLMProvider(ABC):
    provider_name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError