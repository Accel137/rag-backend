from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm.service import LLMService
from app.services.conversation_service import ConversationService
from app.services.chat_service import ChatService



def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


async def get_chat_service(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ChatService:
    return ChatService(
        llm_service=request.app.state.llm_service,
        rag_pipeline=request.app.state.rag_pipeline,
        strategy_pipeline=request.app.state.strategy_pipeline,
        conversation_service=ConversationService(db),
    )
