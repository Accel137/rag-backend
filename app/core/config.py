from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-backend"
    app_env: str = "dev"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag_backend"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change_me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    openai_api_key: str
    qwen_api_key: str
    qwen_base_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

print("QWEN_BASE_URL =", settings.qwen_base_url)
print("QWEN_API_KEY masked =", settings.qwen_api_key[:8], settings.qwen_api_key[-4:])