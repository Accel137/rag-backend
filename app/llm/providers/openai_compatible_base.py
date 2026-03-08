# app/llm/providers/openai_compatible_base.py
from collections.abc import AsyncIterator
from openai import AsyncOpenAI

from app.llm.providers.base import BaseLLMProvider
from app.llm.schemas import LLMRequest, LLMResponse, StreamEvent


class OpenAICompatibleResponsesProvider(BaseLLMProvider):
    def __init__(self, provider_name: str, api_key: str, base_url: str | None, adapter):
        self.provider_name = provider_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.adapter = adapter

    async def generate(self, request: LLMRequest) -> LLMResponse:
        params = self.adapter.build_request_params(request)
        response = await self.client.responses.create(**params)
        return self.adapter.parse_response(response, self.provider_name)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        params = self.adapter.build_request_params(request)
        params["stream"] = True
        stream = await self.client.responses.create(**params)
        async for event in self.adapter.parse_stream(stream):
            yield event