# app/llm/adapters/base.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.llm.schemas import LLMRequest, LLMResponse, StreamEvent


class BaseLLMAdapter(ABC):
    @abstractmethod
    def build_request_params(self, request: LLMRequest) -> dict[str, Any]:
        """把统一请求转成厂商 SDK 需要的参数"""
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: Any, provider_name: str) -> LLMResponse:
        """把厂商响应转成统一响应"""
        raise NotImplementedError

    @abstractmethod
    async def parse_stream(
        self,
        stream: Any,
    ) -> AsyncIterator[StreamEvent]:
        """把厂商流式返回转成统一事件"""
        raise NotImplementedError