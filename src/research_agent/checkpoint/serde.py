"""Checkpoint serializer with a deserialization allow-list.

Checkpoints are read back from an external store. If an attacker (or a bug) can
write bytes into Redis, an unrestricted deserializer turns that into code
execution; langgraph-checkpoint fixed exactly this class of bug in several
releases (e.g. CVE-2025-64439). Graph state in this project holds only
LangChain messages and JSON-native values, so the allow-list can be minimal.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

SAFE_TYPES: tuple[type, ...] = (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)


def allowed_types() -> list[tuple[str, str]]:
    return [(t.__module__, t.__name__) for t in SAFE_TYPES]


def build_serializer(*, strict: bool = True) -> JsonPlusSerializer:
    if not strict:
        return JsonPlusSerializer()
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=cast(Any, allowed_types()),
    )
