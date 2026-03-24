from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, ConversationMessage


class ConversationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_conversation(
        self,
        user_id: str | None = None,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        stmt: Select[tuple[Conversation]] = select(Conversation).where(
            Conversation.id == conversation_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> ConversationMessage:
        message = ConversationMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        self.db.add(message)
        await self.db.flush()
        return message

    async def list_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        stmt: Select[tuple[ConversationMessage]] = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()
        return messages
