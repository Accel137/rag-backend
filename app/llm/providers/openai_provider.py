from app.llm.providers.openai_compatible_base import OpenAICompatibleResponsesProvider
from app.llm.adapters.openai_responses_adapter import OpenAIResponsesAdapter

class OpenAIProvider(OpenAICompatibleResponsesProvider):
    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(
            provider_name="openai",
            api_key=api_key,
            base_url=base_url,
            adapter=OpenAIResponsesAdapter(),
        )
