"""The retrieval pipeline's core: the chain check, the run, the trace.

Toy steps only; the steps that reproduce the real retrieval are tested against the
characterization scenarios (``test_retrieval_characterization.py``).
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from models import ChunkMetadata, RetrievedChunk
from query.context import RetrievalContext, Slot
from query.facts import QueryFactsReader
from query.outcome import Answerable, Declined, DeclineReason
from query.runner import PipelineError, RetrievalPipeline, TraceRecorder
from query.step import (
    AuxRecord,
    Continue,
    Halt,
    RetrievalStep,
    StepResult,
)


def chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=f"chunk {chunk_id}",
        metadata=ChunkMetadata(source_file="a.docx", page_number=None, chunk_index=0),
        score=1.0,
    )


def chunks(*ids: int) -> tuple[RetrievedChunk, ...]:
    return tuple(chunk(i) for i in ids)


def ids(pool) -> list[int]:
    return [c.id for c in pool]


@dataclass
class Toy(RetrievalStep):
    """A step that fills the given slots with fixed passages (or refuses)."""

    name: str
    requires: frozenset[Slot] = frozenset()
    provides: frozenset[Slot] = frozenset()
    fill: dict[str, tuple[RetrievedChunk, ...]] = None  # type: ignore[assignment]
    halt: Declined | None = None
    records: tuple[AuxRecord, ...] = ()
    notes: dict[str, object] = None  # type: ignore[assignment]
    ran: ClassVar[list[str]] = []

    def run(self, context: RetrievalContext) -> StepResult:
        Toy.ran.append(self.name)
        if self.halt is not None:
            return Halt(self.halt, self.records, self.notes or {})
        return Continue(
            context.with_slots(**(self.fill or {})), self.records, self.notes or {}
        )


@pytest.fixture(autouse=True)
def _reset_ran():
    Toy.ran.clear()


def context() -> RetrievalContext:
    return RetrievalContext(facts=QueryFactsReader().read("a question"))


DENSE = frozenset({Slot.DENSE_POOL})
SELECTED = frozenset({Slot.SELECTED})


def search(*pool: int) -> Toy:
    return Toy("search", provides=DENSE, fill={"dense_pool": chunks(*pool)})


def cut(*final: int) -> Toy:
    return Toy(
        "cut", requires=DENSE, provides=SELECTED, fill={"selected": chunks(*final)}
    )


class TestChainCheck:
    def test_a_step_that_requires_a_slot_nothing_provides_is_refused(self):
        with pytest.raises(PipelineError, match="requires"):
            RetrievalPipeline([cut(1)])

    def test_the_order_matters(self):
        with pytest.raises(PipelineError, match="requires"):
            RetrievalPipeline([cut(1), search(1)])

    def test_a_chain_without_the_selected_passages_is_refused(self):
        with pytest.raises(PipelineError, match="selected"):
            RetrievalPipeline([search(1)])

    def test_a_slot_the_caller_supplies_counts_as_provided(self):
        RetrievalPipeline([cut(1)], provided=DENSE)  # does not raise


class TestRun:
    def test_it_runs_the_steps_in_order_and_returns_the_selection(self):
        run = RetrievalPipeline([search(1, 2, 3), cut(2, 1)]).run(context())

        assert Toy.ran == ["search", "cut"]
        assert isinstance(run.outcome, Answerable)
        assert ids(run.outcome.chunks) == [2, 1]
        assert run.halted_by is None

    def test_a_refusal_stops_the_chain_and_names_the_step(self):
        refusal = Declined(DeclineReason.NOT_RELEVANT, stage="gate")
        gate = Toy("gate", requires=DENSE, halt=refusal)

        run = RetrievalPipeline([search(1), gate, cut(1)]).run(context())

        assert Toy.ran == ["search", "gate"]  # "cut" never ran
        assert run.outcome == refusal
        assert run.halted_by == "gate"

    def test_a_step_that_does_not_fill_what_it_declared_is_caught(self):
        liar = Toy("liar", provides=DENSE, fill={})

        with pytest.raises(PipelineError, match="did not fill"):
            RetrievalPipeline([liar, cut(1)]).run(context())


class TestObserver:
    def test_it_is_told_after_each_step_in_order(self):
        recorder = TraceRecorder()

        RetrievalPipeline([search(1, 2), cut(1)]).run(context(), recorder)

        assert [r.step for r in recorder.records] == ["search", "cut"]

    def test_a_record_holds_what_went_in_and_what_came_out(self):
        recorder = TraceRecorder()
        extra = Toy(
            "search",
            provides=DENSE,
            fill={"dense_pool": chunks(1, 2)},
            records=(AuxRecord("year_pool", chunks(9)),),
            notes={"years": [2021]},
        )

        RetrievalPipeline([extra, cut(2)]).run(context(), recorder)

        first, second = recorder.records
        assert ids(first.outputs[Slot.DENSE_POOL]) == [1, 2]
        assert ids(first.aux["year_pool"]) == [9]
        assert first.notes == {"years": [2021]}
        assert ids(second.inputs[Slot.DENSE_POOL]) == [1, 2]
        assert ids(second.outputs[Slot.SELECTED]) == [2]

    def test_a_refusing_step_is_recorded_with_its_refusal_and_no_output(self):
        recorder = TraceRecorder()
        refusal = Declined(DeclineReason.NOT_RELEVANT, stage="gate")
        gate = Toy("gate", requires=DENSE, halt=refusal, notes={"gate_passed": False})

        RetrievalPipeline([search(1), gate, cut(1)]).run(context(), recorder)

        record = recorder.records[-1]
        assert record.declined == refusal
        assert record.outputs == {}
        assert record.notes == {"gate_passed": False}


class TestFacts:
    def test_it_reads_identifiers_and_years_once(self):
        facts = QueryFactsReader().read("costs in 2021 and in case 4.P.20.409/2023/4")

        assert facts.identifiers == ("4.P.20.409/2023/4",)
        assert facts.years == (2021,)  # the 2023 inside the identifier does not count

    def test_a_plain_question_has_none(self):
        facts = QueryFactsReader().read("what did the court decide")

        assert facts.identifiers == ()
        assert facts.years == ()
