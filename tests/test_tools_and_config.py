from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.agent.graph import build_final_answer
from research_agent.agent.state import FinalAnswerDraft, SourceDict
from research_agent.agent.tools import (
    DirectToolExecutor,
    format_sources,
    make_web_search_tool,
)
from research_agent.config import ConfigError, Settings
from tests.conftest import FakeSearchBackend


def _source(i: int) -> SourceDict:
    return SourceDict(
        id=f"S{i}",
        tool="web_search",
        query="q",
        title=f"T{i}",
        location=f"https://e.com/{i}",
        content=f"body {i}",
        score=None,
    )


def test_tool_schema_is_openai_function_format() -> None:
    schema = make_web_search_tool(FakeSearchBackend(), 3).to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert schema["function"]["parameters"]["required"] == ["query"]


def test_executor_validates_arguments() -> None:
    tool = make_web_search_tool(FakeSearchBackend(), 3)
    with pytest.raises(ValidationError):
        DirectToolExecutor().execute(tool, {"query": ""}, thread_id="t", tool_call_id="c")
    hits = DirectToolExecutor().execute(tool, {"query": "ok"}, thread_id="t", tool_call_id="c")
    assert len(hits) == 2


def test_format_sources_includes_ids_and_locations() -> None:
    text = format_sources([_source(1), _source(2)])
    assert "[S1] T1\nhttps://e.com/1" in text
    assert "[S2]" in text


def test_citation_markers_in_text_are_collected_and_deduplicated() -> None:
    draft = FinalAnswerDraft(answer="a [S2] b [S2] c [S1]", short_answer="x")
    final = build_final_answer(draft, [_source(1), _source(2)], hit_step_limit=False)
    assert [c.id for c in final.citations] == ["S2", "S1"]


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("MAX_AGENT_STEPS", "4")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_model == "some-model"
    assert settings.max_agent_steps == 4


def test_missing_secret_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        settings.require_secret("deepseek_api_key")
