# app/llm/adapters/openai_responses_adapter.py
from collections.abc import AsyncIterator
from typing import Any

from app.llm.adapters.base import BaseLLMAdapter
from app.llm.schemas import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    StreamEvent,
    TextOutputItem,
)


class OpenAIResponsesAdapter(BaseLLMAdapter):
    def build_request_params(self, request: LLMRequest) -> dict[str, Any]:
        input_items: list[dict[str, Any]] = []

        for msg in request.messages:
            content = []
            for block in msg.content:
                if block.type == "text":
                    content.append({
                        "type": "input_text",
                        "text": block.text,
                    })

            input_items.append({
                "role": msg.role,
                "content": content,
            })

        params: dict[str, Any] = {
            "model": request.model,
            "input": input_items,
        }

        if request.temperature is not None:
            params["temperature"] = request.temperature

        if request.max_tokens is not None:
            params["max_output_tokens"] = request.max_tokens

        if request.tools:
            params["tools"] = request.tools

        if request.tool_choice is not None:
            params["tool_choice"] = request.tool_choice

        return params

    def parse_response(self, response: Any, provider_name: str) -> LLMResponse:
        texts: list[TextOutputItem] = []

        # 优先走 SDK 常见快捷字段
        output_text = getattr(response, "output_text", None)
        if output_text:
            texts.append(TextOutputItem(text=output_text))
        else:
            # 兜底解析 output items
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for block in getattr(item, "content", []) or []:
                        if getattr(block, "type", None) in {"output_text", "text"}:
                            text_value = getattr(block, "text", None)
                            if text_value:
                                texts.append(TextOutputItem(text=text_value))

        usage = getattr(response, "usage", None)
        parsed_usage = None
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
            parsed_usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        return LLMResponse(
            provider=provider_name,
            model=getattr(response, "model", ""),
            output=texts,
            finish_reason=getattr(response, "status", None),
            usage=parsed_usage,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def parse_stream(self, stream: Any) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                delta = getattr(event, "delta", "")
                if delta:
                    yield StreamEvent(
                        type="text_delta",
                        text=delta,
                        raw=event.model_dump() if hasattr(event, "model_dump") else None,
                    )

            elif event_type in {"response.completed", "response.failed", "response.cancelled"}:
                yield StreamEvent(
                    type="completed",
                    finish_reason=event_type,
                    raw=event.model_dump() if hasattr(event, "model_dump") else None,
                )