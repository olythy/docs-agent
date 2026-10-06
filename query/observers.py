"""Watching a retrieval run: progress lines and the structured audit log.

The steps only *report* (a result and a few notes); they know nothing about log files or
progress output. These observers are told after every step by the pipeline's runner and
do the writing, so a step is never forgotten and never has to care. Retrying a failed
API call is not here either: that stays in the drivers.

Key exports:
    CompositeObserver   -- Forwards to several observers, in order.
    ProgressLogObserver -- One progress line per step.
    AuditLogObserver    -- The structured JSONL events (gate checked, rerank applied).
"""

import logging
from collections.abc import Sequence

from logger import EventLogger, LogAction
from query.context import CHUNK_SLOTS, RetrievalContext
from query.runner import StepObserver
from query.step import Halt, RetrievalStep, StepName, StepResult

_progress = logging.getLogger("query.progress")


class CompositeObserver:
    """Tells every observer about a step, in the order given.

    Args:
        observers: The observers.
    """

    def __init__(self, observers: Sequence[StepObserver]) -> None:
        self._observers = tuple(observers)

    def on_step(
        self,
        step: RetrievalStep,
        before: RetrievalContext,
        result: StepResult,
        seconds: float,
    ) -> None:
        """Forward to every observer."""
        for observer in self._observers:
            observer.on_step(step, before, result, seconds)


class ProgressLogObserver:
    """Logs one line per step (to the logger, never to stdout: the MCP transport uses it)."""

    def on_step(
        self,
        step: RetrievalStep,
        before: RetrievalContext,
        result: StepResult,
        seconds: float,
    ) -> None:
        """Log what the step did."""
        if isinstance(result, Halt):
            _progress.info(
                "[query] %s declined (%s) after %.3fs",
                step.name,
                result.declined.reason,
                seconds,
            )
            return
        sizes = ", ".join(
            f"{slot}={len(value)}"  # type: ignore[arg-type]
            for slot in sorted(step.provides & CHUNK_SLOTS)
            if (value := result.context.value(slot)) is not None
        )
        _progress.info(
            "[query] %s%s (%.3fs)", step.name, f": {sizes}" if sizes else "", seconds
        )


class AuditLogObserver:
    """Writes the structured events the steps' notes stand for.

    Args:
        events: The event log to write to.
    """

    def __init__(self, events: EventLogger) -> None:
        self._events = events

    def on_step(
        self,
        step: RetrievalStep,
        before: RetrievalContext,
        result: StepResult,
        seconds: float,
    ) -> None:
        """Write the event of ``step``, if it has one."""
        if step.name == StepName.RELEVANCE_GATE:
            notes = result.notes
            self._events.log(
                LogAction.RELEVANCE_GATE_CHECKED,
                {
                    "question": before.facts.question,
                    "passed": notes["gate_passed"],
                    "top_score": notes["gate_top_score"],
                    "top_k": notes["gate_depth"],
                    "min_score": notes["gate_min_score"],
                    "candidate_count": notes["gate_candidates"],
                },
            )
