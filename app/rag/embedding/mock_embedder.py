from __future__ import annotations

import random

from app.rag.embedding import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 8):
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._fake_vector(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._fake_vector(query)

    def _fake_vector(self, text: str) -> list[float]:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        return [rng.random() for _ in range(self.dim)]
    
    @property
    def dimension(self) -> int:
        return self.dim
