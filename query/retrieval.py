"""Knowledge base retrieval and answer generation.

This module implements the ``query_knowledge_base`` tool — the second of the two
tools the agent can call (the first being ``add_document`` in ``ingestion/ingest.py``).

Flow:
    1. Embed the user's question with the same driver used during ingestion.
    2. Run a cosine-similarity vector search, widened to a candidate pool.
    3. If nothing in that pool clears the minimum relevance threshold, return
       a polite "I don't know" message — this is the *only* relevance gate,
       and it's deliberately based on pure vector similarity alone (see
       ``_passes_relevance_gate``'s docstring for why), regardless of which
       ``RETRIEVAL_STRATEGY`` is active.
    4. Otherwise, hand the candidate pool to the active
       :class:`RetrievalStrategy` (``settings.RETRIEVAL_STRATEGY`` — a
       Strategy pattern like every other swappable backend in this
       project) to produce the final, ordered top_k: ``hybrid`` (default)
       widens further with a full-text (keyword) search, fuses both ranked
       lists with Reciprocal Rank Fusion, and optionally reranks with a
       cross-encoder; ``vector`` just returns the plain cosine-similarity
       ranking — the pre-hybrid-search behavior, kept as a real, selectable
       strategy (not a special case) so it stays a fair, non-duplicated
       comparison baseline (see ``scripts/evaluate_retrieval.py``).
    5. Pass those chunks to the LLM driver and return its grounded answer.

Usage::

    from query.retrieval import query_knowledge_base
    answer = query_knowledge_base("Mennyi az SZJA tartozásom?")
    print(answer)
"""

import logging
import time
from abc import ABC, abstractmethod

from config import settings
from drivers.embedding import get_embedding_driver
from drivers.llm import get_answer_driver
from drivers.reranker import CrossEncoderRerankerDriver, get_reranker_driver
from logger import LogAction, get_logger
from models import RetrievedChunk
from query.hybrid import reciprocal_rank_fusion
from store import VectorStore, extract_identifier_tokens

# Progress logging, not print(): retrieve_chunks() is called from
# mcp_server.py over an MCP stdio transport, where stray stdout writes can
# corrupt the JSON-RPC protocol stream — confirmed empirically (a print()
# here broke a real client's message parsing mid-call). logging defaults to
# stderr, safe for every caller (CLI scripts, agent.py, mcp_server.py alike).
logger = logging.getLogger(__name__)

# Returned when no chunk clears the relevance threshold.
# Using a constant avoids scatter: every caller sees the same wording,
# and the agent layer (step 6) can test for this exact string if needed.
NO_RESULTS_MESSAGE = (
    "I could not find relevant information about this in the provided documents."
)


class RetrievalStrategy(ABC):
    """Abstract base class for how retrieve_chunks() ranks the final chunks.

    Runs *after* the relevance gate has already passed (see
    ``_passes_relevance_gate``) — that check is deliberately not part of
    this Strategy, since it must stay based on pure vector similarity
    regardless of which strategy is active (see its own docstring for why).
    Selected at runtime via ``settings.RETRIEVAL_STRATEGY``, same shape as
    every other driver/strategy in this project.
    """

    @abstractmethod
    def select_chunks(
        self,
        question: str,
        vector_results: list[RetrievedChunk],
        store: VectorStore,
        top_k: int,
        min_score: float,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Turn an already-fetched vector candidate pool into final chunks.

        Args:
            question: The user's natural-language question.
            vector_results: :meth:`store.VectorStore.search`'s candidate
                pool (fetched with ``min_score=0.0``, so nothing is
                pre-filtered — each strategy decides for itself whether/how
                to use ``min_score``).
            store: The same :class:`store.VectorStore` instance, for
                strategies that need to run further queries (e.g. keyword
                search).
            top_k: Maximum number of chunks to return.
            min_score: ``settings.RETRIEVAL_MIN_SCORE`` (or its override).
            metadata_filter: Optional dict of key-value pairs to restrict
                candidates in secondary searches (e.g. full-text).

        Returns:
            The final, ordered list of at most ``top_k`` chunks.
        """


class VectorRetrievalStrategy(RetrievalStrategy):
    """Pure cosine-similarity ranking — the pre-hybrid-search behavior.

    Kept as a real, selectable strategy (not reimplemented ad hoc) so
    ``scripts/evaluate_retrieval.py`` compares against the actual
    production code path, not a hand-rolled stand-in that could quietly
    drift out of sync with it.
    """

    def select_chunks(
        self,
        question: str,
        vector_results: list[RetrievedChunk],
        store: VectorStore,
        top_k: int,
        min_score: float,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Filter ``vector_results`` by ``min_score`` and truncate to ``top_k``.

        ``vector_results`` is the widened candidate pool, fetched with
        ``min_score=0.0`` — this reapplies the threshold that
        :meth:`store.VectorStore.search` itself would have applied, so this
        strategy's output is identical to calling it directly with
        ``top_k``/``min_score``, not just "whatever's in the wide pool".
        """
        filtered = [c for c in vector_results if c.score >= min_score]
        return filtered[:top_k]


def _apply_top_k_with_guarantees(
    chunks: list[RetrievedChunk], guaranteed_ids: set[int], top_k: int
) -> list[RetrievedChunk]:
    """Truncate ``chunks`` to ``top_k``, but never drop a guaranteed chunk.

    Confirmed live (see docs/decisions.md) that an exact identifier match
    (a case number, ...) surviving the ``RERANKER_MIN_SCORE`` filter still
    isn't enough on its own -- the cross-encoder's relevance *ranking*
    routinely puts it below ``top_k`` other chunks that merely *read* as
    generically on-topic, since the reranker has no notion of "this chunk
    is definitionally correct because its identifier matches." Being in
    the candidate pool only helps if it also survives this final cut.

    Args:
        chunks: Already reranked and score-filtered, in descending score order.
        guaranteed_ids: ``RetrievedChunk.id`` values that must be kept
            regardless of rank (e.g. from :meth:`store.VectorStore.search_by_identifier`).
        top_k: Maximum number of chunks to return.

    Returns:
        Every guaranteed chunk present in ``chunks``, plus the highest-
        scoring remaining chunks up to ``top_k`` total (more than ``top_k``
        only if there are more guaranteed chunks than ``top_k`` itself).
    """
    guaranteed = [c for c in chunks if c.id in guaranteed_ids]
    rest = [c for c in chunks if c.id not in guaranteed_ids]
    return guaranteed + rest[: max(0, top_k - len(guaranteed))]


class HybridRetrievalStrategy(RetrievalStrategy):
    """Vector + keyword search, fused with RRF, then optionally reranked."""

    def __init__(self, reranker_driver_name: str | None = None) -> None:
        """Initialise the strategy.

        Args:
            reranker_driver_name: Explicit reranker driver override, passed
                through to :func:`drivers.reranker.get_reranker_driver`.
                Defaults to ``None``, i.e. read ``settings.RERANKER_DRIVER``.
        """
        self._reranker_driver_name = reranker_driver_name

    def select_chunks(
        self,
        question: str,
        vector_results: list[RetrievedChunk],
        store: VectorStore,
        top_k: int,
        min_score: float,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse ``vector_results`` with a keyword search, then rerank.

        ``min_score`` is intentionally unused here: after RRF fusion (and
        reranking, if enabled), scores live on a scale with no equivalent
        calibrated threshold — see ``_passes_relevance_gate``'s docstring,
        which is exactly why that gate runs on the raw vector results
        instead, before this strategy ever sees them.
        """
        candidate_k = len(vector_results)
        logger.info(
            "[query] Keyword-searching the same candidate pool (%d) ...", candidate_k
        )
        if metadata_filter:
            fulltext_results = store.search_fulltext(
                question, top_k=candidate_k, metadata_filter=metadata_filter
            )
        else:
            fulltext_results = store.search_fulltext(question, top_k=candidate_k)

        fused = reciprocal_rank_fusion(vector_results, fulltext_results)
        logger.info(
            "[query] Fused %d vector + %d keyword result(s) into %d unique candidate(s).",
            len(vector_results),
            len(fulltext_results),
            len(fused),
        )

        # Rescue exact identifiers (case numbers, invoice numbers, ...) that
        # ts_rank's frequency-based scoring loses to common words matching
        # far more often across unrelated documents -- confirmed live via
        # the golden-set eval (see docs/decisions.md). Merged in directly,
        # not RRF-blended: an exact identifier match is a strong enough
        # signal on its own not to need score-averaging with cosine/ts_rank.
        identifier_chunk_ids: set[int] = set()
        identifier_tokens = extract_identifier_tokens(question)
        if identifier_tokens:
            identifier_results = store.search_by_identifier(
                identifier_tokens, top_k=candidate_k
            )
            identifier_chunk_ids = {c.id for c in identifier_results}
            fused_ids = {c.id for c in fused}
            new_matches = [c for c in identifier_results if c.id not in fused_ids]
            if new_matches:
                logger.info(
                    "[query] Identifier tokens %s rescued %d additional candidate(s).",
                    identifier_tokens,
                    len(new_matches),
                )
            fused = new_matches + fused

        reranker = get_reranker_driver(self._reranker_driver_name)
        reranked = reranker.rerank(question, fused)

        if isinstance(reranker, CrossEncoderRerankerDriver):
            threshold = settings.RERANKER_MIN_SCORE
            # An exact identifier match must survive this gate too, not just
            # the later top_k cut (see _apply_top_k_with_guarantees) -- a
            # compound multi-case question can make the cross-encoder score
            # a definitionally-correct chunk (it matches one of several
            # cited cases) well below threshold, since the chunk only reads
            # as on-topic for *part* of the question. Confirmed live via the
            # golden-set eval (see docs/decisions.md).
            valid_chunks = [
                c
                for c in reranked
                if c.score >= threshold or c.id in identifier_chunk_ids
            ]
            get_logger().log(
                LogAction.RERANK_APPLIED,
                {
                    "question": question,
                    "reranker_model": settings.RERANKER_MODEL,
                    "threshold": threshold,
                    "candidates_count": len(reranked),
                    "accepted_count": len(valid_chunks),
                    "top_score": reranked[0].score if reranked else None,
                },
            )
            if not valid_chunks:
                logger.info(
                    "[query] Cross-encoder rejected all chunks (top score %.2f < %.2f threshold).",
                    reranked[0].score if reranked else 0.0,
                    threshold,
                )
                return []
            return _apply_top_k_with_guarantees(
                valid_chunks, identifier_chunk_ids, top_k
            )

        return _apply_top_k_with_guarantees(reranked, identifier_chunk_ids, top_k)


def get_retrieval_strategy() -> RetrievalStrategy:
    """Factory function: return the active retrieval strategy from settings.

    Reads ``settings.RETRIEVAL_STRATEGY`` and instantiates the matching
    strategy.

    Returns:
        A :class:`RetrievalStrategy` instance ready to call.

    Raises:
        ValueError: If ``RETRIEVAL_STRATEGY`` is set to an unknown value.
    """
    strategy_name = settings.RETRIEVAL_STRATEGY.lower()

    if strategy_name == "hybrid":
        return HybridRetrievalStrategy()
    if strategy_name == "vector":
        return VectorRetrievalStrategy()

    raise ValueError(
        f"Unknown RETRIEVAL_STRATEGY: '{strategy_name}'. "
        "Valid options are: 'hybrid', 'vector'."
    )


def retrieve_chunks(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    strategy: RetrievalStrategy | None = None,
    query_vector: list[float] | None = None,
    metadata_filter: dict | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the final context chunks for ``question``.

    Split out from :func:`query_knowledge_base` so retrieval quality can be
    measured directly (see ``scripts/evaluate_retrieval.py``) without
    needing a real LLM call.

    Args:
        question: The user's natural-language question.
        top_k: Override for ``settings.RETRIEVAL_TOP_K``. Maximum chunks to
            return.
        min_score: Override for ``settings.RETRIEVAL_MIN_SCORE``. Similarity
            threshold (0–1) below which the relevance gate fails.
        strategy: Override for ``settings.RETRIEVAL_STRATEGY``. Mainly for
            callers (the eval script) that need to compare strategies
            directly without touching global settings; production callers
            should leave this as ``None`` and configure via ``.env``.
        query_vector: Precomputed embedding for ``question``, skipping both
            the embedding call *and* the embedding driver's lazy model load
            entirely when given.
        metadata_filter: Optional dict of key-value pairs to restrict
            retrieval to matching chunk metadata (JSONB containment).
        store: Optional :class:`store.VectorStore` instance. If omitted,
            instantiates a fresh one.

    Returns:
        The final list of chunks, already ranked/truncated to ``top_k``, or
        ``[]`` if nothing passed the relevance gate (see
        :func:`_passes_relevance_gate`).

    Raises:
        RuntimeError: If the active embedding driver's dimension doesn't
            match the existing document_chunks.embedding column.
    """
    k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
    candidate_k = max(k, settings.RETRIEVAL_CANDIDATE_POOL_SIZE)

    embedding_driver = get_embedding_driver()
    store = store if store is not None else VectorStore()

    with store:
        store.assert_dimension_matches(embedding_driver.dimension)

        if query_vector is None:
            logger.info("[query] Embedding question ...")
            query_vector = embedding_driver.embed_query(question)

        logger.info(
            "[query] Vector-searching a candidate pool of %d chunks ...", candidate_k
        )
        if metadata_filter:
            vector_results = store.search(
                query_vector,
                top_k=candidate_k,
                min_score=0.0,
                metadata_filter=metadata_filter,
            )
        else:
            vector_results = store.search(
                query_vector, top_k=candidate_k, min_score=0.0
            )

        gate_passed = _passes_relevance_gate(
            vector_results, top_k=k, min_score=threshold
        )
        top_score = vector_results[0].score if vector_results else None
        get_logger().log(
            LogAction.RELEVANCE_GATE_CHECKED,
            {
                "question": question,
                "passed": gate_passed,
                "top_score": top_score,
                "top_k": k,
                "min_score": threshold,
                "candidate_count": len(vector_results),
            },
        )
        if not gate_passed:
            logger.info("[query] No relevant chunks found.")
            return []

        active_strategy = strategy if strategy is not None else get_retrieval_strategy()
        logger.info(
            "[query] Selecting final chunks via %s ...", type(active_strategy).__name__
        )
        if metadata_filter:
            chunks = active_strategy.select_chunks(
                question,
                vector_results,
                store,
                top_k=k,
                min_score=threshold,
                metadata_filter=metadata_filter,
            )
        else:
            chunks = active_strategy.select_chunks(
                question,
                vector_results,
                store,
                top_k=k,
                min_score=threshold,
            )

        scores_str = ", ".join(f"{c.score:.4f}" for c in chunks)
        logger.info(
            "[query] Using %d chunk(s) as context. Scores: %s", len(chunks), scores_str
        )
        return chunks


def query_knowledge_base(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    strategy: RetrievalStrategy | None = None,
    metadata_filter: dict | None = None,
    store: VectorStore | None = None,
) -> str:
    """Answer a question using the RAG knowledge base.

    This is the main tool exposed to the agent. It retrieves the most
    relevant document chunks (see :func:`retrieve_chunks`) and generates a
    grounded answer. If nothing clears the relevance gate the agent
    receives an honest "I don't know" response instead of a hallucinated
    answer.

    Args:
        question: The user's natural-language question.
        top_k: Override for ``settings.RETRIEVAL_TOP_K``. Maximum chunks to
            pass to the LLM.
        min_score: Override for ``settings.RETRIEVAL_MIN_SCORE``. Similarity
            threshold (0–1) below which the relevance gate fails.
        strategy: Override for ``settings.RETRIEVAL_STRATEGY``.
        metadata_filter: Optional dict of key-value pairs to restrict
            retrieval to matching chunk metadata (JSONB containment).
        store: Optional :class:`store.VectorStore` instance.

    Returns:
        A string answer grounded in the retrieved chunks, or
        :data:`NO_RESULTS_MESSAGE` if no relevant chunks were found.

    Raises:
        RuntimeError: If the active embedding driver's dimension doesn't
            match the existing document_chunks.embedding column.
    """
    chunks = retrieve_chunks(
        question,
        top_k=top_k,
        min_score=min_score,
        strategy=strategy,
        metadata_filter=metadata_filter,
        store=store,
    )

    if not chunks:
        logger.info("[query] Returning fallback message.")
        return NO_RESULTS_MESSAGE

    logger.info(
        "[query] Generating answer with LLM driver='%s' ...", settings.LLM_DRIVER
    )
    answer_driver = get_answer_driver()
    t0 = time.monotonic()
    answer = answer_driver.answer(question=question, context_chunks=chunks)
    latency = round(time.monotonic() - t0, 3)

    get_logger().log(
        LogAction.ANSWER_GENERATED,
        {
            "question": question,
            "llm_driver": settings.LLM_DRIVER,
            "llm_model": settings.LLM_MODEL,
            "chunk_count": len(chunks),
            "latency_seconds": latency,
        },
    )
    logger.info("[query] Done (%.3fs).", latency)
    return answer


def _passes_relevance_gate(
    vector_results: list[RetrievedChunk], top_k: int, min_score: float
) -> bool:
    """Return True if at least one vector result clears ``min_score``.

    This is the sole "is there a reliable source for this at all" check —
    deliberately based on pure cosine similarity alone, not on the fused
    RRF score or a reranker score. Those live on scales with no natural
    "irrelevant" cutoff (RRF's ``1/(rank+k)`` and a cross-encoder's raw
    logit don't have an equivalent to ``RETRIEVAL_MIN_SCORE``'s calibrated
    0–1 similarity threshold), so reusing this one, already-tuned signal
    as the safety gate — and only *afterwards* letting hybrid search pull
    in additional strong-keyword-but-weak-embedding chunks for context —
    keeps the existing, tested "no reliable source" behavior completely
    unchanged while still gaining hybrid search's recall benefit for chunks
    that already have at least one genuinely relevant anchor nearby.

    Args:
        vector_results: :meth:`store.VectorStore.search`'s candidate pool
            (called with ``min_score=0.0``, so nothing is pre-filtered).
        top_k: Only the top ``top_k`` candidates are checked, matching what
            a plain (non-hybrid) search would have returned.
        min_score: The same threshold :meth:`store.VectorStore.search`
            would have filtered on.

    Returns:
        True if any of the top ``top_k`` vector results has ``score >=
        min_score``.
    """
    return any(c.score >= min_score for c in vector_results[:top_k])
