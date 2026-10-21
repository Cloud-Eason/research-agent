"""Wires real dependencies (DeepSeek + Tavily) into AgentDeps."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from research_agent.agent.graph import AgentDeps
from research_agent.agent.state import FinalAnswerDraft
from research_agent.agent.tools import (
    DirectToolExecutor,
    TavilyBackend,
    ToolExecutor,
    ToolSpec,
    make_web_search_tool,
)
from research_agent.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.require_secret("deepseek_api_key"),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_tools(settings: Settings) -> list[ToolSpec]:
    backend = TavilyBackend(
        api_key=settings.require_secret("tavily_api_key"),
        snippet_chars=settings.web_search_snippet_chars,
    )
    return [make_web_search_tool(backend, settings.web_search_max_results)]


def build_default_deps(
    settings: Settings, executor: ToolExecutor | None = None
) -> AgentDeps:
    llm = build_chat_model(settings)
    tools = build_tools(settings)
    return AgentDeps(
        agent_llm=llm.bind_tools([t.to_openai_schema() for t in tools]),
        # function_calling is the structured-output mode DeepSeek supports reliably;
        # include_raw keeps the AIMessage so token usage can be counted.
        finalizer=llm.with_structured_output(
            FinalAnswerDraft, method="function_calling", include_raw=True
        ),
        tools={t.name: t for t in tools},
        executor=executor or DirectToolExecutor(),
        max_steps=settings.max_agent_steps,
    )