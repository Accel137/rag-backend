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

    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()