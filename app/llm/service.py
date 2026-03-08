from collections.abc import AsyncIterator

from app.llm.registry import LLMProviderRegistry
from app.llm.schemas import LLMRequest, LLMResponse, StreamEvent


class LLMService:
    def __init__(self, registry: LLMProviderRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _validate_request(request: LLMRequest) -> None:
        if not request.model:
            raise ValueError("model is required")
        if not request.messages:
            raise ValueError("messages must not be empty")

    async def generate(
        self,
        request: LLMRequest,
        provider_name: str,
    ) -> LLMResponse:
        self._validate_request(request)
        provider = self.registry.get(provider_name)
        return await provider.generate(request)

    async def stream_generate(
        self,
        request: LLMRequest,
        provider_name: str,
    ) -> AsyncIterator[StreamEvent]:
        self._validate_request(request)
        provider = self.registry.get(provider_name)
        async for event in provider.stream_generate(request):
            yield event