"""Print the model and system fingerprint the LLM API reports right now.

uv run python scripts/served_model.py >> docs/served_model.log

An alias like deepseek-chat can be served by a different backend over time; logging
this before and after an evaluation shows whether the backend stayed the same.
"""

from __future__ import annotations

from datetime import UTC, datetime

from openai import OpenAI

from research_agent.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    client = OpenAI(
        api_key=settings.require_secret("deepseek_api_key"), base_url=settings.llm_base_url
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        max_tokens=5,
        messages=[{"role": "user", "content": "ok"}],
    )
    print(
        f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} requested={settings.llm_model} "
        f"model={response.model} fingerprint={response.system_fingerprint}"
    )
