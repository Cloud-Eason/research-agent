"""Ablation specs (YAML) and run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from research_agent.config import Settings

_SECRET_FIELDS = {"deepseek_api_key", "tavily_api_key", "milvus_token"}
_TRACKED_PACKAGES = (
    "langgraph",
    "langgraph-checkpoint",
    "langchain-openai",
    "pymilvus",
    "sentence-transformers",
    "torch",
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    settings: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None  # use only the first N examples (e.g. the slow reranker)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    layer: Literal["retrieval", "agent"]
    sample_path: Path
    n: int
    base: dict[str, Any]
    runs: list[RunSpec]
    concurrency: int = 4
    top_k: int = 20
    timeout_s: float = 240.0
    offset: int = 0  # skip the first `offset` examples (a held-out dev split of the sample)


def load_ablation(path: Path) -> AblationSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    runs = [
        RunSpec(name=r["name"], settings=r.get("settings") or {}, limit=r.get("limit"))
        for r in raw["runs"]
    ]
    names = [r.name for r in runs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate run names in {path}")
    return AblationSpec(
        name=raw["name"],
        layer=raw["layer"],
        sample_path=Path(raw["dataset"]["sample_path"]),
        n=int(raw["dataset"]["n"]),
        offset=int(raw["dataset"].get("offset", 0)),
        base=raw.get("base") or {},
        runs=runs,
        concurrency=int(raw.get("concurrency", 4)),
        top_k=int(raw.get("top_k", 20)),
        timeout_s=float(raw.get("timeout_s", 240.0)),
    )


def settings_for(base: Settings, spec: AblationSpec, run: RunSpec) -> Settings:
    """Environment settings <- spec.base <- run.settings, validated again."""
    data = base.model_dump()
    data.update(spec.base)
    data.update(run.settings)
    return Settings.model_validate(data)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return out.stdout.strip()
    except OSError:
        return ""


def write_manifest(
    run_dir: Path, spec: AblationSpec, run: RunSpec, settings: Settings, qids: list[str]
) -> None:
    """Everything needed to trace a reported number back to how it was produced."""
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    manifest = {
        "ablation": spec.name,
        "run": run.name,
        "layer": spec.layer,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "n_examples": len(qids),
        "sample_path": str(spec.sample_path),
        "sample_offset": spec.offset,
        "sample_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
        "settings": settings.model_dump(mode="json", exclude=_SECRET_FIELDS),
        "versions": versions,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
