"""Steps that choose the final chunks.

Key exports:
    CosineCutStep           -- The plain similarity cut of the ``vector`` profile.
    TopKWithGuaranteesStep  -- The final cut of the ``hybrid`` profile (with the year quota).
"""

from models import RetrievedChunk
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


def _round_robin_by_document(
    chunks: list[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """Pick up to ``limit`` chunks, taking turns across distinct documents.

    ``chunks`` is in descending score order. Round 1 takes each document's best chunk
    (documents ordered by that chunk's rank), round 2 each document's second best, and
    so on, so a question naming two documents gets both represented before either
    gets a second chunk. Documents are told apart by their source file.
    """
    by_document: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.metadata.source_file, []).append(chunk)

    picked: list[RetrievedChunk] = []
    queues = list(by_document.values())
    while queues and len(picked) < limit:
        for queue in queues:
            if len(picked) == limit:
                break
            picked.append(queue.pop(0))
        queues = [queue for queue in queues if queue]
    return picked


class TopKWithGuaranteesStep(RetrievalStep):
    """Cuts the ranked list to ``top_k``, reserving slots for the question's years.

    * When the question names several documents (``context.spread_documents``), the
      chunks take turns across them first: otherwise the best-scoring document can fill
      every slot and the answer has nothing to say about the others.
    * With ``year_quota`` and years in the question, at least ``ceil(top_k / 2)`` of
      the final chunks come from those years, if there are that many candidates; the
      reranker knows nothing about dates, so widening the pool alone is not enough.
      With ``diversify`` those slots take turns across documents. Soft: with no
      in-period candidate nothing changes.
    * The remaining slots go to the best-ranked chunks.

    Which *documents* the chunks may come from is not decided here: the retrieval is
    handed a store already restricted to them (``query.decision.Scope``).

    Args:
        top_k: The most chunks to keep.
        diversify: Share the year-reserved slots across documents.
        year_quota: Reserve half of the slots for the question's years.
    """

    name = StepName.TOP_K_SELECTION
    requires = frozenset({Slot.RANKED})
    provides = frozenset({Slot.SELECTED})

    def __init__(self, top_k: int, diversify: bool, year_quota: bool) -> None:
        self._top_k = top_k
        self._diversify = diversify
        self._year_quota = year_quota

    def run(self, context: RetrievalContext) -> StepResult:
        chunks = list(context.ranked or ())
        top_k = self._top_k
        years = list(context.facts.years) if self._year_quota else []

        selected = (
            _round_robin_by_document(chunks, top_k) if context.spread_documents else []
        )
        picked = {c.id for c in selected}

        if years:
            wanted = {str(y) for y in years}

            def in_period(chunk: RetrievedChunk) -> bool:
                return (chunk.metadata.document_date or "")[:4] in wanted

            reserve = -(-top_k // 2)
            need = min(
                reserve - sum(1 for c in selected if in_period(c)),
                top_k - len(selected),
            )
            if need > 0:
                candidates = [c for c in chunks if c.id not in picked and in_period(c)]
                extra = (
                    _round_robin_by_document(candidates, need)
                    if self._diversify
                    else candidates[:need]
                )
                selected += extra
                picked |= {c.id for c in extra}

        rest = [c for c in chunks if c.id not in picked]
        final = selected + rest[: max(0, top_k - len(selected))]
        return Continue(context.with_slots(selected=tuple(final)))
