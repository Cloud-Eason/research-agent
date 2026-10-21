"""Application settings, loaded from environment variables and `.env`.

Only Phase 1 fields live here for now; later phases add retrieval, Redis and
eval settings to the same object so there is a single source of configuration.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid at use time."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (DeepSeek, OpenAI-compatible) ---
    deepseek_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    # --- Web search (Tavily) ---
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    web_search_max_results: int = Field(default=5, ge=1, le=20)
    web_search_snippet_chars: int = Field(default=800, ge=100)

    # --- Agent loop ---
    max_agent_steps: int = Field(default=8, ge=1, le=50)

    def require_secret(self, name: str) -> str:
        """Return a secret's value or fail fast with a readable error.

        Secrets default to empty so that tests and offline tooling can build a
        Settings object; code paths that actually call external APIs use this.
        """
        value = getattr(self, name)
        assert isinstance(value, SecretStr)
        secret = value.get_secret_value()
        if not secret:
            raise ConfigError(f"{name.upper()} is not set (check your .env file)")
        return secret


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
