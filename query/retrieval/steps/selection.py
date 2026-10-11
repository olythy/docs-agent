"""Steps that choose the final chunks.

Key exports:
    TopKWithGuaranteesStep  -- The final cut of the ``best_chunks`` profile (with the year quota).
    DocumentExpansionStep   -- Reads the whole of the one document that clearly leads.
    OpeningChunkStep        -- Adds the first chunk of the one document a question names.
"""

from models import RetrievedChunk
from query.retrieval.context import RetrievalContext, Slot
from query.retrieval.documents import document_scores
from query.retrieval.step import Continue, RetrievalStep, StepName, StepResult
from store import VectorStore


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


class DocumentExpansionStep(RetrievalStep):
    """Replaces the cut by the whole of the document that clearly leads the ranking.

    A question that describes a decision (its court, year and subject) is answered from one
    document, yet the final cut takes the best few chunks of several: the passage that
    answers it is often a lower-ranked chunk of the right document (measured: for 75% of the
    questions it was in the context). When one document leads the others by ``margin``
    (its score is at least ``margin`` times the second's), its chunks are read in full
    (up to ``max_chunks``), the best-ranked first; otherwise the cut stays as it is.

    Args:
        store: The chunk store (restricted to the scope).
        margin: How many times the second document's score the leader must have.
        max_chunks: The most chunks of the document to read.
    """

    name = StepName.DOCUMENT_EXPANSION
    requires = frozenset({Slot.RANKED, Slot.SELECTED})
    provides = frozenset({Slot.SELECTED})

    def __init__(self, store: VectorStore, margin: float, max_chunks: int) -> None:
        self._store = store
        self._margin = margin
        self._max_chunks = max_chunks

    def run(self, context: RetrievalContext) -> StepResult:
        scores = document_scores(context.ranked or ())
        if not scores:
            return Continue(context)
        leader, leader_score = scores[0]
        if len(scores) > 1 and leader_score < self._margin * scores[1][1]:
            return Continue(context)
        whole = self._store.chunks_of_source(leader, self._max_chunks)
        if not whole:
            return Continue(context)
        by_id = {chunk.id: chunk for chunk in whole}
        ranked_first = [c for c in (context.ranked or ()) if c.id in by_id]
        seen = {c.id for c in ranked_first}
        rest = [c for c in whole if c.id not in seen]
        # best-ranked chunks first (the prompt puts the first ones last, nearest the question)
        ordered = [by_id[c.id] for c in ranked_first] + rest
        return Continue(context.with_slots(selected=tuple(ordered[: self._max_chunks])))


class OpeningChunkStep(RetrievalStep):
    """Adds the document's first chunk when the question names one document and lacks it.

    The opening of a court decision holds the parties, the subject of the case and the
    ruling, which is what a question about one named case usually asks. The search ranks
    passages by their similarity to the question, and a long reasoning section can outrank
    the opening, so the answer would have to rest on the generated summary alone. With one
    document in scope the opening is added as the last-ranked chunk (the context grows by
    one); every other question is untouched.

    Args:
        store: The chunk store, already restricted to the document in scope.
    """

    name = StepName.OPENING_CHUNK
    requires = frozenset({Slot.SELECTED})
    provides = frozenset({Slot.SELECTED})

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, context: RetrievalContext) -> StepResult:
        selected = tuple(context.selected or ())
        if not context.single_document or not selected:
            return Continue(context)
        if any(c.metadata.chunk_index == 0 for c in selected):
            return Continue(context)
        opening = self._store.opening_chunks(limit=1)
        if not opening:
            return Continue(context)
        return Continue(context.with_slots(selected=(*selected, opening[0])))
