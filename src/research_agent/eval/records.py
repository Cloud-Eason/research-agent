"""JSONL helpers shared by the runners and the report."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def latest_records(path: Path) -> list[dict[str, Any]]:
    """One record per qid: the last successful one, else the last error."""
    best: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path):
        prev = best.get(rec["qid"])
        if prev is None or "error" not in rec or "error" in prev:
            best[rec["qid"]] = rec
    return list(best.values())


def completed_qids(path: Path) -> set[str]:
    """qids with a successful record; errored ones are retried on the next run."""
    return {r["qid"] for r in latest_records(path) if "error" not in r}
