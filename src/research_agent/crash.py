"""Deterministic crash injection for the crash-recovery demo and tests.

    CRASH_AT=tool:2   kill the process right after the 2nd tool result is cached
    CRASH_AT=agent:2  kill it right after the 2nd LLM call returns (before its
                      checkpoint is written)

The default handler is os._exit, which skips `finally` blocks, atexit hooks and
buffered I/O flushing, the closest an in-process hook gets to `kill -9`. Tests
swap in a handler that raises instead, so pytest itself survives.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

CRASH_EXIT_CODE = 137  # what a shell reports for a SIGKILLed process

_counts: dict[str, int] = {}


def _hard_exit(event: str) -> None:
    sys.stderr.write(f"\n[crash] simulated crash at {event} -> os._exit({CRASH_EXIT_CODE})\n")
    sys.stderr.flush()
    os._exit(CRASH_EXIT_CODE)


_handler: Callable[[str], None] = _hard_exit


def set_handler(handler: Callable[[str], None] | None) -> None:
    """Replace the crash action (tests); None restores os._exit."""
    global _handler
    _handler = handler or _hard_exit
    _counts.clear()


def maybe_crash(event: str) -> None:
    spec = os.environ.get("CRASH_AT", "")
    if not spec:
        return
    name, _, nth = spec.partition(":")
    if name != event:
        return
    _counts[event] = _counts.get(event, 0) + 1
    if _counts[event] == int(nth or "1"):
        _handler(f"{event}:{_counts[event]}")
