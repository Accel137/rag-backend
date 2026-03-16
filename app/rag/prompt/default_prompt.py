from __future__ import annotations

from app.llm.schemas import LLMMessage, TextContentBlock
from app.rag.prompt.base import BasePrompt
from app.rag.schemas import RetrievedChunk


DEFAULT_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question "
    "using the provided context when it is relevant. If the context is "
    "insufficient, say so clearly and avoid fabricating facts."
)


class DefaultPrompt(BasePrompt):
    prompt_name = "default"

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No retrieved context."

        sections: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            header_parts = [f"[{index}]"]

            document_id = metadata.get("document_id")
            if document_id:
                header_parts.append(f"document_id={document_id}")

            source = metadata.get("source")
            if source:
                header_parts.append(f"source={source}")

            chunk_index = metadata.get("chunk_index")
            if chunk_index is not None:
                header_parts.append(f"chunk_index={chunk_index}")

            score = f"{chunk.score:.4f}"
            header = " ".join(header_parts)
            sections.append(f"{header} score={score}\n{chunk.text}")

        return "\n\n".join(sections)

    def build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        question = question.strip()
        context = self.build_context(chunks)

        return (
            "Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

    def build_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[LLMMessage]:
        return [
            LLMMessage(
                role="system",
                content=[TextContentBlock(text=self.system_prompt)],
            ),
            LLMMessage(
                role="user",
                content=[TextContentBlock(text=self.build_user_prompt(question, chunks))],
            ),
        ]
