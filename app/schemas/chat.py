from typing import Literal
from pydantic import BaseModel 
from typing import Optional

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    knowledge_base_id: Optional[str] = None


class ChatOutputText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    output: list[ChatOutputText]
    model: str
    provider: str
    finish_reason: str | None = None
    usage: ChatUsage | None = None
