
from fastapi import Request
from app.llm.service import LLMService
from app.services.chat_service import ChatService



def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service