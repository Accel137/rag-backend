from openai import OpenAI

from app.rag.embedding import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # 预设维度（也可以写成字典映射）
        if model == "text-embedding-3-small":
            self._dimension = 1536
        elif model == "text-embedding-3-large":
            self._dimension = 3072
        else:
            raise ValueError(f"Unknown embedding model: {model}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[query],
        )

        return response.data[0].embedding