from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./internmate.db"
    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    embedding_provider: str = "hash"  # "gemini" or "hash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    job_max_age_days: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
