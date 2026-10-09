"""Running a chain of retrieval steps, and watching it run.

:class:`RetrievalPipeline` checks the chain when it is built (every slot a step requires
is provided by an earlier one, the chain ends with the selected chunks) and runs it,
stopping at the first step that refuses. It does not log or measure anything itself:
after every step it tells an observer, and the observers (:class:`TraceRecorder` here,
logging ones elsewhere) do. Retrying a failed API call stays in the drivers.

Key exports:
    PipelineError    -- A chain that cannot work, or a step that broke its contract.
    PipelineRun      -- How a run ended.
    RetrievalPipeline  -- The validated chain.
    StepObserver     -- Told after each step.
    StageRecord, TraceRecorder -- What a step held going in and coming out.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from models import RetrievedChunk
from query.outcome import Answerable, Declined
from query.retrieval.context import CHUNK_SLOTS, RetrievalContext, Slot
from query.retrieval.step import Halt, RetrievalStep, StepResult


class PipelineError(RuntimeError):
    """The chain is invalid, or a step did not do what it declared."""


@dataclass(frozen=True)
class PipelineRun:
    """How a run ended.

    Attributes:
        outcome: The chunks, or the refusal.
        context: The context as the last step that ran left it.
        halted_by: The name of the step that refused, if one did.
    """

    outcome: Answerable | Declined
    context: RetrievalContext
    halted_by: str | None = None


class StepObserver(Protocol):
    """Told after each step ran."""

    def on_step(
        self,
        step: RetrievalStep,
        before: RetrievalContext,
        result: StepResult,
        seconds: float,
    ) -> None:
        """Called with the step, the context it got, what it returned and how long it took."""
        ...


class RetrievalPipeline:
    """A validated chain of steps.

    Args:
        steps: The steps, in order.
        provided: Slots already filled when the run starts (e.g. a query vector the
            caller supplies).

    Raises:
        PipelineError: If a step requires a slot no earlier step provides, or no step
            provides the selected chunks.
    """

    def __init__(
        self,
        steps: Sequence[RetrievalStep],
        provided: frozenset[Slot] = frozenset(),
    ) -> None:
        available = set(provided)
        for step in steps:
            missing = step.requires - available
            if missing:
                raise PipelineError(
                    f"step {step.name!r} requires {sorted(missing)} "
                    "but no earlier step provides it"
                )
            available |= step.provides
        if Slot.SELECTED not in available:
            raise PipelineError("no step provides the selected chunks")
        self._steps = tuple(steps)

    def run(
        self, context: RetrievalContext, observer: StepObserver | None = None
    ) -> PipelineRun:
        """Run the chain on ``context`` until it ends or a step refuses.

        Raises:
            PipelineError: If a step continues without filling a slot it provides.
        """
        for step in self._steps:
            started = time.monotonic()
            result = step.run(context)
            if observer is not None:
                observer.on_step(step, context, result, time.monotonic() - started)
            if isinstance(result, Halt):
                return PipelineRun(result.declined, context, halted_by=step.name)
            unfilled = [s for s in step.provides if result.context.value(s) is None]
            if unfilled:
                raise PipelineError(
                    f"step {step.name!r} did not fill {sorted(unfilled)}"
                )
            context = result.context
        selected = context.selected
        assert selected is not None  # guaranteed by the validation and the check above
        return PipelineRun(Answerable(selected), context)


@dataclass(frozen=True)
class StageRecord:
    """What one step held going in and coming out, for the trace.

    Attributes:
        step: The step's name.
        inputs: The chunk slots it required, as they were.
        outputs: The chunk slots it provided, as it left them (empty if it refused).
        aux: Its side results, by label.
        notes: Its small facts.
        seconds: How long it took.
        declined: The refusal, if it refused.
    """

    step: str
    inputs: dict[Slot, tuple[RetrievedChunk, ...]]
    outputs: dict[Slot, tuple[RetrievedChunk, ...]]
    aux: dict[str, tuple[RetrievedChunk, ...]]
    notes: dict[str, object]
    seconds: float
    declined: Declined | None = None


@dataclass
class TraceRecorder:
    """An observer that keeps one :class:`StageRecord` per step that ran."""

    records: list[StageRecord] = field(default_factory=list)

    def on_step(
        self,
        step: RetrievalStep,
        before: RetrievalContext,
        result: StepResult,
        seconds: float,
    ) -> None:
        """Record the step (see :class:`StepObserver`)."""
        inputs = {
            slot: value
            for slot in step.requires & CHUNK_SLOTS
            if (value := before.value(slot)) is not None
        }
        halted = isinstance(result, Halt)
        outputs = (
            {}
            if halted
            else {
                slot: value
                for slot in step.provides & CHUNK_SLOTS
                if (value := result.context.value(slot)) is not None  # type: ignore[union-attr]
            }
        )
        self.records.append(
            StageRecord(
                step=step.name,
                inputs=inputs,  # type: ignore[arg-type]
                outputs=outputs,  # type: ignore[arg-type]
                aux={r.label: r.chunks for r in result.records},
                notes=dict(result.notes),
                seconds=seconds,
                declined=result.declined if isinstance(result, Halt) else None,
            )
        )
