from collections.abc import AsyncIterator
import json

from app.schemas.chat import ChatRequest, ChatResponse, ChatOutputText, ChatUsage
from app.schemas.chat import ChatMessage
from app.llm.schemas import (
    LLMRequest,
    LLMMessage,
    TextContentBlock,
    LLMResponse,
)
from app.llm.service import LLMService
from app.rag.pipeline import RAGPipeline
from app.rag.strategy import StrategyPipeline
from app.services.conversation_service import ConversationService


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        rag_pipeline: RAGPipeline,
        strategy_pipeline: StrategyPipeline,
        conversation_service: ConversationService,
    ):
        self.llm_service = llm_service
        self.rag_pipeline = rag_pipeline
        self.strategy_pipeline = strategy_pipeline
        self.conversation_service = conversation_service

    def _build_llm_request(
        self,
        req: ChatRequest,
        messages: list[LLMMessage],
    ) -> LLMRequest:
        return LLMRequest(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

    @staticmethod
    def _to_chat_messages(messages: list) -> list[ChatMessage]:
        return [
            ChatMessage(role=message.role, content=message.content)
            for message in messages
        ]

    @staticmethod
    def _to_llm_messages(messages: list[ChatMessage]) -> list[LLMMessage]:
        return [
            LLMMessage(
                role=msg.role,
                content=[TextContentBlock(text=msg.content)],
            )
            for msg in messages
        ]

    async def _build_messages_for_request(
        self,
        req: ChatRequest,
        messages: list[ChatMessage],
    ) -> list[LLMMessage]:
        if not req.knowledge_base_id:
            return self._to_llm_messages(messages)

        plan = await self.strategy_pipeline.decide(messages)
        if not plan.query:
            return self._to_llm_messages(messages)

        chunks = await self.rag_pipeline.retrieve(
            query=plan.query,
            strategy=plan.strategy,
            filters={"knowledge_base_id": req.knowledge_base_id},
        )
        rag_messages = self.rag_pipeline.build_messages(
            query=plan.query,
            strategy=plan.strategy,
            chunks=chunks,
        )
        history = self._to_llm_messages(messages[:-1])
        return [rag_messages[0], *history, rag_messages[1]]

    @staticmethod
    def _extract_text(llm_resp: LLMResponse) -> str:
        return "".join(
            item.text
            for item in llm_resp.output
            if item.type == "text"
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        conversation = await self.conversation_service.get_or_create_conversation(
            conversation_id=req.conversation_id,
        )
        await self.conversation_service.add_message(
            conversation_id=conversation.id,
            role="user",
            content=req.message,
        )
        history = await self.conversation_service.list_recent_messages(
            conversation_id=conversation.id,
        )
        chat_messages = self._to_chat_messages(history)
        messages = await self._build_messages_for_request(req, chat_messages)
        llm_req = self._build_llm_request(req, messages=messages)
        llm_resp: LLMResponse = await self.llm_service.generate(
            request=llm_req,
            provider_name=req.provider,
        )

        text = self._extract_text(llm_resp)
        await self.conversation_service.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=text,
        )

        usage = None
        if llm_resp.usage:
            usage = ChatUsage(
                input_tokens=llm_resp.usage.input_tokens,
                output_tokens=llm_resp.usage.output_tokens,
                total_tokens=llm_resp.usage.total_tokens,
            )

        return ChatResponse(
            conversation_id=conversation.id,
            content=text,
            output=[
                ChatOutputText(text=item.text)
                for item in llm_resp.output
                if item.type == "text"
            ],
            model=llm_resp.model,
            provider=llm_resp.provider,
            finish_reason=llm_resp.finish_reason,
            usage=usage,
        )

    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[str]:
        conversation = await self.conversation_service.get_or_create_conversation(
            conversation_id=req.conversation_id,
        )
        await self.conversation_service.add_message(
            conversation_id=conversation.id,
            role="user",
            content=req.message,
        )
        history = await self.conversation_service.list_recent_messages(
            conversation_id=conversation.id,
        )
        chat_messages = self._to_chat_messages(history)
        messages = await self._build_messages_for_request(req, chat_messages)
        llm_req = self._build_llm_request(req, messages=messages)
        text_parts: list[str] = []

        conversation_payload = {
            "type": "conversation",
            "conversation_id": conversation.id,
            "done": False,
        }
        yield f"data: {json.dumps(conversation_payload, ensure_ascii=False)}\n\n"

        async for event in self.llm_service.stream_generate(
            request=llm_req,
            provider_name=req.provider,
        ):
            if event.type == "text_delta":
                delta = event.text or ""
                if delta:
                    text_parts.append(delta)
                payload = {
                    "type": "text_delta",
                    "text": delta,
                    "done": False,
                    "finish_reason": None,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            elif event.type == "completed":
                full_text = "".join(text_parts)
                await self.conversation_service.add_message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=full_text,
                )
                payload = {
                    "type": "completed",
                    "text": full_text,
                    "done": True,
                    "finish_reason": event.finish_reason,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            elif event.type == "error":
                payload = {
                    "type": "error",
                    "error": event.error,
                    "done": True,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"
