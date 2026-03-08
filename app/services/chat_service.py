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


class ChatService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_llm_request(self, req: ChatRequest) -> LLMRequest:
        return LLMRequest(
            model=req.model,
            messages=[
                LLMMessage(
                    role=msg.role,
                    content=[TextContentBlock(text=msg.content)],
                )
                for msg in req.messages
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

    @staticmethod
    def _extract_text(llm_resp: LLMResponse) -> str:
        return "".join(
            item.text
            for item in llm_resp.output
            if item.type == "text"
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        llm_req = self._build_llm_request(req)
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
        llm_req = self._build_llm_request(req)

        async for event in self.llm_service.stream_generate(
            request=llm_req,
            provider_name=req.provider,
        ):
            if event.type == "text_delta":
                payload = {
                    "type": "text_delta",
                    "delta": event.text or "",
                    "done": False,
                    "finish_reason": None,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            elif event.type == "completed":
                payload = {
                    "type": "completed",
                    "delta": "",
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