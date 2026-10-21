"""Application settings, loaded from environment variables and `.env`.

One Settings object for the whole project; ablation configs (Phase 4) override
individual fields rather than defining a second configuration system.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    # bge v1.5 recommends this prefix for short queries against passages.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, ge=1)

    # --- Chunking (measured in the embedding model's own tokens) ---
    chunk_size: int = Field(default=256, ge=16)
    chunk_overlap: int = Field(default=32, ge=0)
    chunk_prepend_title: bool = True

    # --- Milvus ---
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr = Field(default=SecretStr(""))
    kb_collection_prefix: str = "kb"
    kb_collection: str | None = None  # explicit name overrides the derived one
    bm25_analyzer: str = "standard"  # "chinese" for Chinese corpora

    # --- Agent tools (routing ablations switch these) ---
    enable_web_search: bool = True
    enable_kb_search: bool = True

    # --- Hybrid retrieval: every stage can be switched off for ablations ---
    retrieval_use_dense: bool = True
    retrieval_use_bm25: bool = True
    retrieval_use_rerank: bool = True
    retrieval_dense_top_k: int = Field(default=20, ge=1, le=200)
    retrieval_bm25_top_k: int = Field(default=20, ge=1, le=200)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_rerank_candidates: int = Field(default=20, ge=1, le=200)
    retrieval_final_top_n: int = Field(default=5, ge=1, le=50)
    # Drop reranked hits below this score; None keeps everything.
    retrieval_min_rerank_score: float | None = None

    # --- Reranker (cross-encoder) ---
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_max_length: int = Field(default=512, ge=32)

    # --- Persistence (Phase 3) ---
    checkpointer: Literal["memory", "redis"] = "redis"
    redis_url: str = "redis://localhost:6379/0"
    checkpoint_ttl_s: int = Field(default=86_400, ge=60)  # sliding, refreshed on every write
    checkpoint_keep_last: int = Field(default=20, ge=1)
    tool_cache_ttl_s: int = Field(default=86_400, ge=60)

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not (self.retrieval_use_dense or self.retrieval_use_bm25):
            raise ValueError("enable at least one of retrieval_use_dense / retrieval_use_bm25")
        return self

    def require_secret(self, name: str) -> str:
        """Return a secret's value or fail fast with a readable error."""
        value = getattr(self, name)
        assert isinstance(value, SecretStr)
        secret = value.get_secret_value()
        if not secret:
            raise ConfigError(f"{name.upper()} is not set (check your .env file)")
        return secret


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
