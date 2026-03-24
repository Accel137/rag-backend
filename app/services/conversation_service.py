from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, ConversationMessage
from app.repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repository = ConversationRepository(db)

    async def get_or_create_conversation(
        self,
        conversation_id: str | None = None,
        user_id: str | None = None,
        title: str | None = None,
    ) -> Conversation:
        if conversation_id:
            conversation = await self.repository.get_conversation(conversation_id)
            if conversation is not None:
                return conversation

        conversation = await self.repository.create_conversation(
            user_id=user_id,
            title=title,
        )
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> ConversationMessage:
        message = await self.repository.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def list_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        return await self.repository.list_recent_messages(
            conversation_id=conversation_id,
            limit=limit,
        )
