"""Steps that can end the retrieval with a refusal.

Key exports:
    RelevanceGateStep -- Refuses when nothing is similar enough to the question.
"""

from query.context import RetrievalContext, Slot
from query.outcome import Declined, DeclineReason
from query.step import Continue, Halt, RetrievalStep, StepName, StepResult


class RelevanceGateStep(RetrievalStep):
    """Refuses unless one of the best candidates is similar enough to the question.

    The one "is there a reliable source at all" check, deliberately on the raw cosine
    similarity: the fused and reranked scores have no calibrated "irrelevant" cutoff.
    It looks at the top ``depth`` candidates of the pool as the search returned them.

    Args:
        depth: How many of the best candidates to look at.
        min_score: The cosine similarity one of them has to reach.
    """

    name = StepName.RELEVANCE_GATE
    requires = frozenset({Slot.DENSE_POOL})

    def __init__(self, depth: int, min_score: float) -> None:
        self._depth = depth
        self._min_score = min_score

    def run(self, context: RetrievalContext) -> StepResult:
        pool = context.dense_pool or ()
        passed = any(c.score >= self._min_score for c in pool[: self._depth])
        notes = {
            "gate_passed": passed,
            "gate_top_score": pool[0].score if pool else None,
            "gate_min_score": self._min_score,
            "gate_depth": self._depth,
            "gate_candidates": len(pool),
        }
        if passed:
            return Continue(context, notes=notes)
        return Halt(
            Declined(DeclineReason.NOT_RELEVANT, stage=str(self.name)), notes=notes
        )
