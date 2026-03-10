from pydantic import BaseModel, Field
from typing import Optional, Dict

class DocumentMetadata(BaseModel):
    document_id: str
    source: str
    title: str | None = None
    knowledge_base_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class ChunkMetadata(DocumentMetadata):
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None
    page: int | None = None
    section: str | None = None

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata


class IngestRequest(BaseModel):
    document_id: str
    text: str
    metadata: Dict[str, str] = {}


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, str] = {}


class RAGQuery(BaseModel):
    query: str
    top_k: int = 5


class RAGResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]



