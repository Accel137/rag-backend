from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class APPSettings(BaseSettings):
    app_name: str = "rag-backend"
    app_env: str = "dev"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    jwt_secret_key: str = "change_me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60



class DBSettings(BaseSettings):
    database_url: str
    redis_url: str


class LLMSettings(BaseSettings):
    openai_api_key: str
    qwen_api_key: str
    qwen_base_url: str


class RAGSettings(BaseSettings):
    # simple, langchaian
    chunk_provider: str = "simple"
    chunk_size: int = 500
    chunk_overlap: int = 100

    # mock, openai
    embedding_provider: str = "mock"


    # mock: mock
    # openai: text-embedding-3-small
    embedding_model: str = "mock"

    # memory
    vector_store_provider: str = "memory"

    openai_api_key: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


    database_url: str
    redis_url: str

    openai_api_key: str
    qwen_api_key: str
    qwen_base_url: str

    @property

    def app(self) -> APPSettings:
        return APPSettings()


    @property
    def db(self) -> DBSettings:
        return DBSettings(
            database_url=self.database_url, 
            redis_url=self.redis_url
        )
    
    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            openai_api_key=self.openai_api_key,
            qwen_api_key=self.qwen_api_key,
            qwen_base_url=self.qwen_base_url
        )
    
    @property
    def rag(self) -> RAGSettings:
        return RAGSettings(
            chunk_provider="simple",
            chunk_size=500,
            chunk_overlap=100,
            embedding_provider="mock",
            embedding_model="mock",
            vector_store_provider="memory",
            openai_api_key=self.openai_api_key
        )

settings = Settings()
