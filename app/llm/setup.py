
from app.core.config import settings
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.qwen_provider import QwenProvider
# from app.llm.providers.claude_provider import ClaudeProvider
from app.llm.registry import LLMProviderRegistry
from app.llm.service import LLMService

def build_llm_registry() -> LLMProviderRegistry:
    registry = LLMProviderRegistry()

    if settings.openai_api_key:
        registry.register(
            OpenAIProvider(
                api_key=settings.openai_api_key
            )
        )

    if settings.qwen_api_key and settings.qwen_base_url:
        registry.register(
            QwenProvider(
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
        )

    # if settings.CLAUDE_API_KEY:
    #     registry.register(
    #         ClaudeProvider(
    #             api_key=settings.CLAUDE_API_KEY,
    #         )
    #     )

    return registry


def build_llm_service() -> LLMService:
    registry = build_llm_registry()
    return LLMService(registry)
