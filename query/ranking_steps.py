"""Steps that fuse and reorder candidates.

Key exports:
    RrfFusionStep -- Fuses the vector and the keyword candidates (Reciprocal Rank Fusion).
    RerankStep    -- Reorders the candidates with a reranker driver.
    ListwiseRerankStep -- The optional final pass in which a language model picks the
                          document that answers the question.
"""

from collections.abc import Callable

from drivers.reranker import RerankerDriver
from models import RetrievedChunk
from query.context import RetrievalContext, Slot
from query.hybrid import reciprocal_rank_fusion
from query.step import Continue, RetrievalStep, StepName, StepResult


class RrfFusionStep(RetrievalStep):
    """Fuses the vector and the keyword candidates into one ranked list.

    Only rank positions count (the two scores live on different scales); the vector
    list is first, so on a tie its chunk object is the one kept.
    """

    name = StepName.RRF_FUSION
    requires = frozenset({Slot.DENSE_POOL, Slot.KEYWORD_POOL})
    provides = frozenset({Slot.RANKED})

    def run(self, context: RetrievalContext) -> StepResult:
        fused = reciprocal_rank_fusion(
            list(context.dense_pool or ()), list(context.keyword_pool or ())
        )
        return Continue(context.with_slots(ranked=tuple(fused)))


class RerankStep(RetrievalStep):
    """Reorders the whole ranked list (pinned chunks included) with a reranker.

    Args:
        reranker: The reranker driver.
    """

    name = StepName.RERANK
    requires = frozenset({Slot.RANKED})
    provides = frozenset({Slot.RANKED})

    def __init__(self, reranker: RerankerDriver) -> None:
        self._reranker = reranker

    def run(self, context: RetrievalContext) -> StepResult:
        reranked = self._reranker.rerank(
            context.facts.question, list(context.ranked or ())
        )
        return Continue(context.with_slots(ranked=tuple(reranked)))


#: Reorders chunks for a question (see :func:`query.listwise_rerank.listwise_rerank`).
ListwiseRanker = Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]]


class ListwiseRerankStep(RetrievalStep):
    """Runs the optional listwise pass on the ranked list, just before the final cut.

    It costs one language-model call per question, which is why a profile includes it
    only when it is switched on. Only the order changes.

    Args:
        ranker: The listwise reranker, with its model and limits already bound.
    """

    name = StepName.LISTWISE_RERANK
    requires = frozenset({Slot.RANKED})
    provides = frozenset({Slot.RANKED})

    def __init__(self, ranker: ListwiseRanker) -> None:
        self._ranker = ranker

    def run(self, context: RetrievalContext) -> StepResult:
        ranked = self._ranker(context.facts.question, list(context.ranked or ()))
        return Continue(context.with_slots(ranked=tuple(ranked)))
