"""Deterministic fakes so graph tests need no API keys or network."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from research_agent.agent.graph import AgentDeps
from research_agent.agent.state import FinalAnswerDraft, SearchHit
from research_agent.agent.tools import DirectToolExecutor, make_web_search_tool


def tool_call(query: str, call_id: str, name: str = "web_search") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


class ScriptedChatModel:
    """Returns pre-scripted messages in order and records every prompt."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: list[BaseMessage], /) -> Any:
        self.calls.append(list(input))
        if not self._responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return self._responses.pop(0)


class LoopingToolCaller:
    """Always asks for another search; used to exercise the step limit."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.calls = 0

    def invoke(self, input: list[BaseMessage], /) -> AIMessage:
        self.calls += 1
        n = next(self._ids)
        return tool_call(f"query {n}", f"loop_{n}")


class FakeSearchBackend:
    def __init__(self, hits_per_query: int = 2, fail: bool = False) -> None:
        self.hits_per_query = hits_per_query
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("search backend down")
        return [
            SearchHit(
                title=f"{query} result {i}",
                location=f"https://example.com/{len(self.queries)}/{i}",
                content=f"Content about {query} #{i}",
            )
            for i in range(min(self.hits_per_query, max_results))
        ]


@pytest.fixture
def make_deps() -> Callable[..., AgentDeps]:
    def _make(
        agent_llm: Any,
        draft: FinalAnswerDraft | list[FinalAnswerDraft] | None = None,
        backend: FakeSearchBackend | None = None,
        max_steps: int = 5,
    ) -> AgentDeps:
        drafts = draft if isinstance(draft, list) else [
            draft or FinalAnswerDraft(answer="done", short_answer="done")
        ]
        backend = backend or FakeSearchBackend()
        tool = make_web_search_tool(backend, max_results=5)
        return AgentDeps(
            agent_llm=agent_llm,
            finalizer=ScriptedChatModel(list(drafts)),
            tools={tool.name: tool},
            executor=DirectToolExecutor(),
            max_steps=max_steps,
            today=lambda: date(2026, 1, 1),
        )

    return _make
