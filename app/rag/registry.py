from app.rag.retriever import BaseRetriever
from app.rag.prompt import BasePrompt


class RetrieverRegistry:
    def __init__(self) -> None:
        self._retrievers: dict[str, BaseRetriever] = {}

    def register(self, retriever: BaseRetriever) -> None:
        if retriever.retriever_name in self._retrievers:
            raise ValueError(f"Retriever already registered: {retriever.retriever_name}")
        self._retrievers[retriever.retriever_name] = retriever

    def get(self, retriever_name: str) -> BaseRetriever:
        retriever = self._retrievers.get(retriever_name)
        if retriever is None:
            raise ValueError(f"Unknown retriever: {retriever_name}")
        return retriever


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: dict[str, BasePrompt] = {}

    def register(self, prompt: BasePrompt) -> None:
        if prompt.prompt_name in self._prompts:
            raise ValueError(f"Prompt already registered: {prompt.prompt_name}")
        self._prompts[prompt.prompt_name] = prompt

    def get(self, prompt_name: str) -> BasePrompt:
        prompt = self._prompts.get(prompt_name)
        if prompt is None:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        return prompt
