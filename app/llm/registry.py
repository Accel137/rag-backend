from app.llm.providers.base import BaseLLMProvider


class LLMProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}

    def register(self, provider: BaseLLMProvider) -> None:
        if provider.provider_name in self._providers:
            raise ValueError(f"Provider already registered: {provider.provider_name}")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> BaseLLMProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider

    def has_provider(self, provider_name: str) -> bool:
        return provider_name in self._providers

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

