# app/llm/schemas.py
from typing import Any, Literal
from pydantic import BaseModel, Field


class TextContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[TextContentBlock]


class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TextOutputItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


class LLMResponse(BaseModel):
    provider: str
    model: str
    output: list[TextOutputItem]
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    raw: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    type: Literal["text_delta", "completed", "error"]
    text: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class LLMRequest(BaseModel):
    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)