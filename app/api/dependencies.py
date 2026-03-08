from functools import lru_cache
from fastapi import Depends
from app.llm.registry import LLMProviderRegistry
from app.llm.service import LLMService
from app.llm.setup import build_llm_registry
from app.services.chat_service import ChatService


@lru_cache
def get_llm_registry() -> LLMProviderRegistry:
    return build_llm_registry()


@lru_cache
def get_llm_service() -> LLMService:
    registry = get_llm_registry()
    return LLMService(registry)

@lru_cache
def get_chat_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> ChatService:
    return ChatService(llm_service)