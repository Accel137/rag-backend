from __future__ import annotations

from dataclasses import dataclass

from app.rag.schemas import RAGStrategy


@dataclass
class RAGPlan:
    query: str
    strategy: RAGStrategy
