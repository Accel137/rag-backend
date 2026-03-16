
from app.core.config import LLMSettings
from app.llm.providers.openai_provider import OpenAIProvider
from app.llm.providers.qwen_provider import QwenProvider
# from app.llm.providers.claude_provider import ClaudeProvider
from app.llm.registry import LLMProviderRegistry
from app.llm.service import LLMService

def build_llm_registry(config: LLMSettings) -> LLMProviderRegistry:
    registry = LLMProviderRegistry()

    if config.openai_api_key:
        registry.register(
            OpenAIProvider(
                api_key=config.openai_api_key
            )
        )

    if config.qwen_api_key and config.qwen_base_url:
        registry.register(
            QwenProvider(
                api_key=config.qwen_api_key,
                base_url=config.qwen_base_url,
            )
        )

    return registry


def build_llm_service(config: LLMSettings) -> LLMService:
    registry = build_llm_registry(config)
    return LLMService(registry)
