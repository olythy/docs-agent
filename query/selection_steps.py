"""Steps that choose the final chunks.

Key exports:
    CosineCutStep -- The plain similarity cut of the ``vector`` profile.
"""

from query.context import RetrievalContext, Slot
from query.step import Continue, RetrievalStep, StepName, StepResult


class CosineCutStep(RetrievalStep):
    """Keeps the candidates at or above a similarity threshold, best first, up to ``top_k``.

    Args:
        min_score: The cosine similarity threshold.
        top_k: The most chunks to keep.
    """

    name = StepName.COSINE_CUT
    requires = frozenset({Slot.DENSE_POOL})
    provides = frozenset({Slot.SELECTED})

    def __init__(self, min_score: float, top_k: int) -> None:
        self._min_score = min_score
        self._top_k = top_k

    def run(self, context: RetrievalContext) -> StepResult:
        kept = [c for c in (context.dense_pool or ()) if c.score >= self._min_score]
        return Continue(context.with_slots(selected=tuple(kept[: self._top_k])))
