"""Builds the persistence layer from Settings."""

from __future__ import annotations

from typing import Any

import redis
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent.tools import ToolExecutor
from research_agent.checkpoint.redis_saver import RedisCheckpointSaver
from research_agent.checkpoint.tool_cache import CachedToolExecutor
from research_agent.config import Settings


def redis_client(settings: Settings) -> redis.Redis:
    client = redis.Redis.from_url(settings.redis_url)
    client.ping()  # fail fast with a clear error instead of on the first checkpoint
    return client


def build_persistence(
    settings: Settings,
) -> tuple[BaseCheckpointSaver[Any], ToolExecutor | None]:
    """(checkpointer, tool executor). The memory option keeps Phase 1 behaviour."""
    if settings.checkpointer == "memory":
        return InMemorySaver(), None
    client = redis_client(settings)
    saver = RedisCheckpointSaver(
        client, ttl_s=settings.checkpoint_ttl_s, keep_last=settings.checkpoint_keep_last
    )
    return saver, CachedToolExecutor(client, ttl_s=settings.tool_cache_ttl_s)
