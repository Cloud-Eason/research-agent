"""Idempotent tool execution: results cached in Redis per tool call.

Why this is needed on top of LangGraph's pending writes: pending writes are
recorded per *node*, and our tools node runs every tool call of one AI message
before it returns. If the process dies after call 1 of 2, the node produced no
writes yet, so on resume LangGraph re-runs the whole node. This cache turns that
re-run into cache hits, so no external API is called twice.

The tool_call_id is stable across a resume because it lives in the AIMessage
that was already checkpointed. The key also includes a hash of the tool name and
arguments: some providers number tool calls per response ("call_0", ...), so the
id alone could collide across turns of one thread and return another query's
results.

Failures are not cached: a tool that raised is retried on resume.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from research_agent import crash
from research_agent.agent.state import SearchHit
from research_agent.agent.tools import DirectToolExecutor, ToolExecutor, ToolSpec
from research_agent.checkpoint import keys


def cache_entry(spec_name: str, args: dict[str, Any], tool_call_id: str) -> str:
    digest = hashlib.sha1(
        json.dumps([spec_name, args], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    return f"{tool_call_id}:{digest}"


class CachedToolExecutor:
    def __init__(
        self, client: redis.Redis, *, ttl_s: int = 86_400, inner: ToolExecutor | None = None
    ) -> None:
        self.client: Any = client
        self.ttl_s = ttl_s
        self.inner: ToolExecutor = inner or DirectToolExecutor()
        self.hits = 0
        self.misses = 0

    def execute(
        self, spec: ToolSpec, args: dict[str, Any], *, thread_id: str, tool_call_id: str
    ) -> list[SearchHit]:
        entry = cache_entry(spec.name, args, tool_call_id)
        key = keys.tool_cache_key(thread_id, entry)
        cached = self.client.get(key)
        if cached is not None:
            self.hits += 1
            return [SearchHit.model_validate(item) for item in json.loads(cached)]

        hits = self.inner.execute(spec, args, thread_id=thread_id, tool_call_id=tool_call_id)
        self.misses += 1
        index = keys.tool_cache_index_key(thread_id)
        pipe = self.client.pipeline(transaction=True)
        pipe.set(key, json.dumps([h.model_dump() for h in hits]), ex=self.ttl_s)
        pipe.sadd(index, entry)
        pipe.expire(index, self.ttl_s)
        pipe.execute()
        # The result is durable now; a crash here must not cause a second API call.
        crash.maybe_crash("tool")
        return hits
