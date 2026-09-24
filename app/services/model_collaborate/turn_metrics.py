"""
Per-turn, per-stage latency and token accounting.

The readiness gate adds a Gemini call to every turn and removes the whole
generation pipeline from some of them. Whether that trade is worth it is a
measurement, not an opinion, so each stage's wall clock and token usage are
recorded and logged once per turn.

Carried in a ContextVar rather than passed down the call chain: GeminiService
is the only place that sees a response's usage metadata, and it is five layers
below the code that knows which stage is running. A ContextVar is also the
right scope -- asyncio copies the context per task, so the concurrent phase-2
calls each inherit the same recorder without racing for it, and a turn that is
cancelled simply stops recording.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_current: ContextVar["TurnMetrics | None"] = ContextVar("turn_metrics", default=None)
_current_stage: ContextVar[str | None] = ContextVar("turn_metrics_stage", default=None)


@dataclass
class StageMetrics:
    """One pipeline stage's totals. `calls` > 1 where a stage retries (the response gate)."""

    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class TurnMetrics:
    """Every stage of one turn, in the order the stages first ran."""

    decision: str | None = None
    stages: dict[str, StageMetrics] = field(default_factory=dict)

    def stage(self, name: str) -> StageMetrics:
        return self.stages.setdefault(name, StageMetrics())

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.stages.values())

    def format(self) -> str:
        """
        Renders the turn as one log line.

        Parameters:
        - none

        Returns:
        - str: "decision=respond total=1843tok stage=readiness 0.41s 612tok(560+52) ..." — goes to the log line ModelCollaborateService emits per turn. Wall clock per stage is summed across that stage's calls, so a concurrent phase shows each call's own duration rather than the elapsed time of the group; gather logs the elapsed figure separately.
        """
        parts = [
            f"decision={self.decision or 'unknown'}",
            f"total={self.total_tokens}tok",
        ]
        for name, s in self.stages.items():
            parts.append(
                f"{name} {s.seconds:.2f}s "
                f"{s.total_tokens}tok({s.prompt_tokens}+{s.output_tokens})"
                + (f" x{s.calls}" if s.calls > 1 else "")
            )
        return " | ".join(parts)


@contextmanager
def turn():
    """
    Opens a recorder for one chat turn.

    Parameters:
    - none

    Returns:
    - TurnMetrics: the recorder, also installed as the current one for the duration — used by ModelCollaborateService.model_orchestration, which logs it when the turn ends

    An already-open recorder is reused rather than shadowed, so a caller that
    wraps a whole turn to read its totals (a benchmark, a test) gets the same
    object the pipeline writes into instead of an empty one.
    """
    existing = _current.get()
    if existing is not None:
        yield existing
        return

    metrics = TurnMetrics()
    token = _current.set(metrics)
    try:
        yield metrics
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str):
    """
    Attributes everything Gemini does inside the block to one named stage.

    Parameters:
    - name (str): the stage label, e.g. "readiness", "topics", "example", "grounding", "reply", "gate", "split" — comes from the call site

    Returns:
    - None: yields nothing. Outside a turn() it is a no-op, so probe scripts and tests can call the services directly without setting anything up.
    """
    metrics = _current.get()
    if metrics is None:
        yield
        return
    token = _current_stage.set(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        entry = metrics.stage(name)
        entry.calls += 1
        entry.seconds += time.perf_counter() - started
        _current_stage.reset(token)



def record_usage(prompt_tokens: int, output_tokens: int) -> None:
    """
    Adds one Gemini call's token usage to whichever stage is currently open.

    Parameters:
    - prompt_tokens (int): tokens sent — comes from GeminiService, out of the response's usage metadata
    - output_tokens (int): tokens generated — comes from GeminiService

    Returns:
    - None: accumulates onto the current stage, or does nothing outside a turn()/stage() pair. Never raises: usage metadata is a reporting nicety, and a turn must not fail because a response did not carry it.
    """
    metrics = _current.get()
    name = _current_stage.get()
    if metrics is None or name is None:
        return
    entry = metrics.stage(name)
    entry.prompt_tokens += prompt_tokens
    entry.output_tokens += output_tokens


def set_decision(decision: str) -> None:
    """
    Records the readiness decision this turn took.

    Parameters:
    - decision (str): "respond", "wait" or "no_reply" — comes from ModelCollaborateService.model_orchestration

    Returns:
    - None: stores it on the current recorder, or does nothing outside a turn()
    """
    metrics = _current.get()
    if metrics is not None:
        metrics.decision = decision
