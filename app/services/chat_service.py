from collections.abc import AsyncIterator
import json

from app.schemas.chat import ChatRequest, ChatResponse, ChatOutputText, ChatUsage
from app.llm.schemas import (
    LLMRequest,
    LLMMessage,
    TextContentBlock,
    LLMResponse,
)
from app.llm.service import LLMService
from app.rag.pipeline import RAGPipeline
from app.rag.strategy import StrategyPipeline


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        rag_pipeline: RAGPipeline,
        strategy_pipeline: StrategyPipeline,
    ):
        self.llm_service = llm_service
        self.rag_pipeline = rag_pipeline
        self.strategy_pipeline = strategy_pipeline

    def _build_llm_request(
        self,
        req: ChatRequest,
        messages: list[LLMMessage] | None = None,
    ) -> LLMRequest:
        return LLMRequest(
            model=req.model,
            messages=messages
            or [
                LLMMessage(
                    role=msg.role,
                    content=[TextContentBlock(text=msg.content)],
                )
                for msg in req.messages
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

    async def _build_messages_for_request(self, req: ChatRequest) -> list[LLMMessage]:
        if not req.knowledge_base_id:
            return [
                LLMMessage(
                    role=msg.role,
                    content=[TextContentBlock(text=msg.content)],
                )
                for msg in req.messages
            ]

        plan = await self.strategy_pipeline.decide(req.messages)
        if not plan.query:
            return [
                LLMMessage(
                    role=msg.role,
                    content=[TextContentBlock(text=msg.content)],
                )
                for msg in req.messages
            ]

        chunks = await self.rag_pipeline.retrieve(
            query=plan.query,
            strategy=plan.strategy,
            filters={"knowledge_base_id": req.knowledge_base_id},
        )
        return self.rag_pipeline.build_messages(
            query=plan.query,
            strategy=plan.strategy,
            chunks=chunks,
        )

    @staticmethod
    def _extract_text(llm_resp: LLMResponse) -> str:
        return "".join(
            item.text
            for item in llm_resp.output
            if item.type == "text"
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        messages = await self._build_messages_for_request(req)
        llm_req = self._build_llm_request(req, messages=messages)
        llm_resp: LLMResponse = await self.llm_service.generate(
            request=llm_req,
            provider_name=req.provider,
        )

        text = self._extract_text(llm_resp)

        usage = None
        if llm_resp.usage:
            usage = ChatUsage(
                input_tokens=llm_resp.usage.input_tokens,
                output_tokens=llm_resp.usage.output_tokens,
                total_tokens=llm_resp.usage.total_tokens,
            )

        return ChatResponse(
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
        messages = await self._build_messages_for_request(req)
        llm_req = self._build_llm_request(req, messages=messages)

        async for event in self.llm_service.stream_generate(
            request=llm_req,
            provider_name=req.provider,
        ):
            if event.type == "text_delta":
                payload = {
                    "type": "text_delta",
                    "text": event.text or "",
                    "done": False,
                    "finish_reason": None,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            elif event.type == "completed":
                payload = {
                    "type": "completed",
                    "text": "",
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
