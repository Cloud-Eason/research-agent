"""Rebuild summary tables from existing results without re-running anything.

uv run python scripts/report.py results/retrieval_v1
uv run python scripts/report.py results/agent_v1 --baseline dense
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_agent.eval.report import write_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--baseline", default=None, help="run to compare others against")
    args = parser.parse_args()
    out = write_reports(args.eval_dir, baseline=args.baseline)
    print(out.read_text(encoding="utf-8"))
