"""Today's trace keys, reproduced from the new step records.

The diagnostics (``funnel``, ``retrieval-snapshot``, the characterization tests) read a
:class:`models.RetrievalTrace` with fixed stage names. Those names carry history, and
some of them are not quite what they say. The new pipeline records one uniform
:class:`query.runner.StageRecord` per step; this class is the one place that maps them
back onto the old keys, quirks included, so the old contract stays byte-identical while
the pipeline itself stays clean. It is deliberate, isolated debt: when the keys are
renamed (``listwise`` is really "the list entering the final cut"), only this file and
the baseline change.

The quirks, all pinned by the characterization tests:

* ``fused`` is what the identifier step left, written whether or not the question had
  an identifier (and ``identifier`` only when it had);
* ``listwise`` is the list entering the final cut, written whether or not the listwise
  step is in the chain, and absent when the reranker's score gate refused;
* ``final`` is empty when a step after the relevance gate refused, and absent when the
  relevance gate itself refused.

Key exports:
    LegacyTraceProjection -- Turns step records into a :class:`models.RetrievalTrace`.
"""

from collections.abc import Sequence

from models import RetrievalTrace
from query.context import Slot
from query.runner import PipelineRun, StageRecord
from query.step import StepName

#: The step notes the original trace had. Steps report more (for the logs); only these
#: reach the old trace, so its contract stays unchanged.
_LEGACY_NOTES = frozenset({"gate_passed", "gate_top_score", "gate_min_score", "years"})


class LegacyTraceProjection:
    """Maps step records onto the original ``RetrievalTrace`` stage names."""

    def project(
        self, records: Sequence[StageRecord], run: PipelineRun
    ) -> RetrievalTrace:
        """Build the trace the original pipeline would have recorded.

        Args:
            records: The records of the steps that ran, in order.
            run: How the run ended.
        """
        by_step = {r.step: r for r in records}
        trace = RetrievalTrace()

        def output(step: StepName, slot: Slot) -> tuple | None:
            record = by_step.get(step)
            return record.outputs.get(slot) if record else None

        def aux(step: StepName, label: str) -> tuple | None:
            record = by_step.get(step)
            return record.aux.get(label) if record else None

        def put(stage: str, chunks: tuple | None) -> None:
            if chunks is not None:
                trace.record(stage, list(chunks))

        put("vector", output(StepName.DENSE_SEARCH, Slot.DENSE_POOL))
        put("vector_years", aux(StepName.YEAR_DENSE_WIDENING, "year_pool"))
        put("vector_csls", output(StepName.CSLS_REORDER, Slot.DENSE_POOL))
        put("fulltext", output(StepName.KEYWORD_SEARCH, Slot.KEYWORD_POOL))
        put("fulltext_years", aux(StepName.YEAR_KEYWORD_WIDENING, "year_pool"))
        put("identifier", aux(StepName.IDENTIFIER_PIN, "identifier_matches"))
        put("fused", output(StepName.IDENTIFIER_PIN, Slot.RANKED))
        put("reranked", output(StepName.RERANK, Slot.RANKED))

        selection = by_step.get(StepName.TOP_K_SELECTION) or by_step.get(
            StepName.COSINE_CUT
        )
        if selection is not None:
            put("listwise", selection.inputs.get(Slot.RANKED))
            put("final", selection.outputs.get(Slot.SELECTED))
        elif run.halted_by is not None and run.halted_by != StepName.RELEVANCE_GATE:
            trace.record("final", [])

        for record in records:
            trace.notes.update(
                {k: v for k, v in record.notes.items() if k in _LEGACY_NOTES}
            )
        return trace
