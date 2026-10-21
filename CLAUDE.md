# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLM research agent: a hand-built LangGraph ReAct loop (DeepSeek via the OpenAI-compatible API) with two retrieval tools — `web_search` (Tavily) and `kb_search` (hybrid RAG over Milvus) — Redis-backed crash recovery, and (planned) evaluation/ablations. Work is organized in phases referenced throughout docstrings: Phase 1 agent loop, Phase 2 RAG, Phase 3 persistence/crash recovery, Phase 4 eval + ablations (`src/research_agent/eval/` and `configs/ablations/` are still empty).

## Commands

Package manager is `uv` (Python 3.11; CPU-only torch on Linux via a custom index in `pyproject.toml`).

```bash
uv sync                                   # install (incl. dev group)
docker compose up -d                      # Milvus standalone (+etcd, minio), Redis, Attu UI on :8000
uv run python scripts/check_env.py        # verify Redis/Milvus/API keys are reachable

uv run pytest                             # unit tests (integration excluded by default via addopts)
uv run pytest tests/test_graph.py::test_name   # single test
uv run pytest -m integration              # needs Docker Milvus/Redis running
uv run ruff check . && uv run ruff format .
uv run mypy src                           # strict mode

uv run research-agent ask "question" -v   # also: chat, resume --thread-id X; --no-web/--no-kb/--max-steps
uv run python scripts/ingest.py data/papers/ [--chunk-size 512 --overlap 64 --recreate]
uv run python scripts/search_kb.py "query" [--no-rerank|--no-dense] --k 3   # inspect each retrieval stage
uv run python scripts/crash_recovery_demo.py --offline [--crash-at agent:2] # needs Redis only
```

Config comes from `.env` / env vars via `research_agent.config.Settings` (pydantic-settings; field name = upper-cased env var, e.g. `CHECKPOINTER=memory` to run without Redis, `HF_HUB_OFFLINE=1`). `get_settings()` is `lru_cache`d — override per run with `settings.model_copy(update=...)`, as the CLI does. Ablations are meant to toggle existing `Settings` fields (`enable_*`, `retrieval_use_*`, chunk sizes), not introduce a second config system.

## Architecture

**Agent graph** (`agent/graph.py`): explicit `StateGraph` `start → agent ⇄ tools → finalize`. Each node is a checkpoint boundary. `start` resets per-turn counters; `route_after_agent` sends to `finalize` when there are no tool calls or `max_agent_steps` is used up; `finalize` patches dangling tool calls with stub `ToolMessage`s (keeps history valid for the OpenAI-style API), calls a structured-output finalizer (`FinalAnswerDraft`, `method="function_calling"` for DeepSeek), drops citations not pointing at retrieved sources, and replaces the agent's draft message by reusing its id. `recursion_limit_for()` must stay in sync with the graph shape.

**Dependencies are injected** via `AgentDeps` (`agent/factory.py` wires real ones). The LLM is a `ChatModel` protocol, so tests use `ScriptedChatModel` / `FakeSearchBackend` from `tests/conftest.py` — no keys or network needed for unit tests.

**Tools** are `ToolSpec`s (not LangChain `@tool`) and are executed by the graph's `tools` node through a `ToolExecutor`. Every tool returns `SearchHit`s, which the tools node numbers as sources `S1, S2, …`; `sources` accumulates across turns of a thread so ids never collide, and the LLM cites `[S#]`.

**State rule** (`agent/state.py`): graph state holds only LangChain messages and JSON-native values (TypedDicts/dicts). Pydantic models are used only at boundaries. This is required because `checkpoint/serde.py` deserializes with a strict allow-list of message types — adding a non-JSON type to state will break Redis checkpoint loading.

**Persistence** (`checkpoint/`): `build_persistence()` returns `(checkpointer, executor)` — `InMemorySaver` + direct execution, or the custom `RedisCheckpointSaver` + `CachedToolExecutor`. The Redis saver stores channel values as versioned blobs (writes only changed channels), keeps the newest `checkpoint_keep_last` checkpoints with blob GC, uses sliding TTL, and hash-tags keys by thread id (`keys.py` documents the layout). Async methods delegate to sync via `asyncio.to_thread`. The tool cache makes resume idempotent: key = `tool_call_id` + hash of (tool name, args); failures aren't cached. Graphs must be invoked with `durability="sync"` (`cli.invoke_durably`) — otherwise a lost checkpoint makes the LLM re-emit new tool_call_ids and the cache misses. Resume = `graph.invoke(None, config)`.

**Crash injection** (`crash.py`): `CRASH_AT=tool:N` / `agent:N` triggers `os._exit(137)` at hooks in the tool cache and agent node; tests swap the handler with `crash.set_handler`.

**RAG** (`rag/`): loaders → `RecursiveChunker` (sized in the embedding model's tokens, optional title prefix) → bge-small embeddings → `MilvusStore` (one collection with dense HNSW/COSINE vectors plus server-side BM25 sparse vectors via a Milvus Function). Collection name is derived from embedding model + chunk config (`kb_bge_small_en_v1_5_c256_o32`), so chunk-size ablations switch collections rather than re-ingest. `HybridRetriever`: dense + BM25 → RRF → cross-encoder rerank (bge-reranker-base); every stage is switchable and results include intermediate rankings for per-stage Recall@k. `build_tools` imports the RAG factory lazily because it loads models.
