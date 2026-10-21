"""Retrieval tools exposed to the agent.

Tools are described by a `ToolSpec` (name, description, args schema, run
function) instead of LangChain `@tool` objects. Reasons:

* The graph's tools node executes calls itself, so Phase 3 can wrap execution
  with an idempotency cache keyed by (thread_id, tool_call_id).
* Every retrieval tool returns `SearchHit`s, so web_search and kb_search
  (Phase 2) share one source-numbering and citation path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from research_agent.agent.state import SearchHit, SourceDict
from research_agent.rag.models import ScoredChunk
from research_agent.rag.retrieval import RetrievalResult


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class TavilyBackend:
    """Web search via the Tavily API."""

    def __init__(self, api_key: str, snippet_chars: int = 800) -> None:
        from tavily import TavilyClient  # imported lazily: optional at test time

        self._client = TavilyClient(api_key=api_key)
        self._snippet_chars = snippet_chars

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def search(self, query: str, max_results: int) -> list[SearchHit]:
        response: dict[str, Any] = self._client.search(
            query=query, max_results=max_results, search_depth="basic"
        )
        hits: list[SearchHit] = []
        for item in response.get("results", []):
            url = str(item.get("url", ""))
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or url),
                    location=url,
                    content=str(item.get("content") or "")[: self._snippet_chars],
                    score=item.get("score"),
                )
            )
        return hits


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any], list[SearchHit]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class WebSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description="A focused search query covering one sub-question.",
    )


def make_web_search_tool(backend: SearchBackend, max_results: int) -> ToolSpec:
    def _run(args: WebSearchArgs) -> list[SearchHit]:
        return backend.search(args.query, max_results)

    return ToolSpec(
        name="web_search",
        description=(
            "Search the public web. Use for recent events, facts not in the local "
            "knowledge base, or when other tools return nothing useful. Returns "
            "numbered sources you can cite."
        ),
        args_model=WebSearchArgs,
        run=_run,
    )


class KbSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description="A focused query covering one sub-question; keywords and names help.",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Number of passages to return.")


class KbRetriever(Protocol):
    def retrieve(self, query: str, top_n: int | None = None) -> RetrievalResult: ...


def kb_hit_to_search_hit(hit: ScoredChunk) -> SearchHit:
    chunk = hit.chunk
    page = f" (p.{chunk.page})" if chunk.page is not None else ""
    location = f"{chunk.source}#page={chunk.page}" if chunk.page is not None else chunk.source
    return SearchHit(
        title=f"{chunk.title}{page}",
        location=location,
        content=chunk.text,
        score=hit.score,
        ref=chunk.chunk_id,
    )


def make_kb_search_tool(retriever: KbRetriever) -> ToolSpec:
    def _run(args: KbSearchArgs) -> list[SearchHit]:
        result = retriever.retrieve(args.query, top_n=args.top_k)
        return [kb_hit_to_search_hit(h) for h in result.hits]

    return ToolSpec(
        name="kb_search",
        description=(
            "Search the local knowledge base of ingested documents (papers, reports, "
            "reference pages) with hybrid keyword + semantic retrieval. Returns numbered "
            "passages you can cite."
        ),
        args_model=KbSearchArgs,
        run=_run,
    )


class ToolExecutor(Protocol):
    """Executes one tool call. Phase 3 adds a Redis-cached implementation."""

    def execute(
        self, spec: ToolSpec, args: dict[str, Any], *, thread_id: str, tool_call_id: str
    ) -> list[SearchHit]: ...


class DirectToolExecutor:
    """Validates arguments and calls the tool, with no caching."""

    def execute(
        self, spec: ToolSpec, args: dict[str, Any], *, thread_id: str, tool_call_id: str
    ) -> list[SearchHit]:
        parsed = spec.args_model.model_validate(args)
        return spec.run(parsed)


def format_sources(sources: Sequence[SourceDict]) -> str:
    """Render sources as the text the LLM sees in tool results and prompts."""
    blocks = [f"[{s['id']}] {s['title']}\n{s['location']}\n{s['content']}" for s in sources]
    return "\n\n".join(blocks)
