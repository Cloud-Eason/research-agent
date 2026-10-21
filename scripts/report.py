"""Rebuild summary tables from existing results without re-running anything.

uv run python scripts/report.py results/retrieval_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

from research_agent.eval.report import write_reports

if __name__ == "__main__":
    out = write_reports(Path(sys.argv[1]))
    print(out.read_text(encoding="utf-8"))
