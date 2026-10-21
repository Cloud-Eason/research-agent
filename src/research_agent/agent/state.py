"""Graph state and answer schemas.

Design rule: everything stored in graph state is either a LangChain message or
a JSON-native value (TypedDict / dict / str / int). Pydantic models are used
only at the boundaries (LLM structured output, CLI, eval). This keeps
checkpoints serializable under a strict deserialization allow-list, which
matters in Phase 3 when state is read back from Redis.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A raw result from a retrieval tool, before it gets a source id."""

    title: str
    location: str  # URL for web results; e.g. "paper.pdf#p3" for KB results
    content: str
    score: float | None = None


class SourceDict(TypedDict):
    """A retrieved item as stored in state. `id` is stable within a thread."""

    id: str  # "S1", "S2", ...
    tool: str
    query: str
    title: str
    location: str
    content: str
    score: float | None


class FinalAnswerDraft(BaseModel):
    """Schema the LLM fills in during the finalize step."""

    answer: str = Field(
        description=(
            "Complete answer to the user's latest question, with inline citation "
            "markers like [S1] after each supported claim."
        )
    )
    short_answer: str = Field(
        description=(
            "The minimal answer span: an entity, name, date, number, or 'yes'/'no'. "
            "No full sentences, no citation markers. Use 'unknown' if the evidence "
            "is insufficient."
        )
    )
    cited_source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the sources (e.g. ['S1', 'S3']) that support the answer.",
    )


class Citation(BaseModel):
    id: str
    title: str
    location: str


class FinalAnswer(BaseModel):
    answer: str
    short_answer: str
    citations: list[Citation]
    invalid_citation_ids: list[str] = Field(default_factory=list)
    hit_step_limit: bool = False


def merge_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    """Reducer that sums token counters key by key."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = merged.get(key, 0) + value
    return merged


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Accumulates across turns of a thread, so source ids never collide.
    sources: Annotated[list[SourceDict], operator.add]
    # Per-turn counters, reset by the start node.
    step: int
    hit_step_limit: bool
    final: dict[str, Any] | None  # FinalAnswer.model_dump()
    # Thread-level token accounting (one thread per question in eval).
    token_usage: Annotated[dict[str, int], merge_usage]
