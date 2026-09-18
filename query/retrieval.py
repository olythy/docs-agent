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
from abc import ABC, abstractmethod

from config import settings
from drivers.embedding import get_embedding_driver
from drivers.llm import get_answer_driver
from drivers.reranker import get_reranker_driver
from query.hybrid import reciprocal_rank_fusion
from store import VectorStore

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
        vector_results: list[dict],
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> list[dict]:
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
        vector_results: list[dict],
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> list[dict]:
        """Filter ``vector_results`` by ``min_score`` and truncate to ``top_k``.

        ``vector_results`` is the widened candidate pool, fetched with
        ``min_score=0.0`` — this reapplies the threshold that
        :meth:`store.VectorStore.search` itself would have applied, so this
        strategy's output is identical to calling it directly with
        ``top_k``/``min_score``, not just "whatever's in the wide pool".
        """
        filtered = [c for c in vector_results if c["score"] >= min_score]
        return filtered[:top_k]


class HybridRetrievalStrategy(RetrievalStrategy):
    """Vector + keyword search, fused with RRF, then optionally reranked."""

    def select_chunks(
        self,
        question: str,
        vector_results: list[dict],
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> list[dict]:
        """Fuse ``vector_results`` with a keyword search, then rerank.

        ``min_score`` is intentionally unused here: after RRF fusion (and
        reranking, if enabled), scores live on a scale with no equivalent
        calibrated threshold — see ``_passes_relevance_gate``'s docstring,
        which is exactly why that gate runs on the raw vector results
        instead, before this strategy ever sees them.
        """
        candidate_k = len(vector_results)
        logger.info("[query] Keyword-searching the same candidate pool (%d) ...", candidate_k)
        fulltext_results = store.search_fulltext(question, top_k=candidate_k)

        fused = reciprocal_rank_fusion(vector_results, fulltext_results)
        logger.info(
            "[query] Fused %d vector + %d keyword result(s) into %d unique candidate(s).",
            len(vector_results),
            len(fulltext_results),
            len(fused),
        )

        reranker = get_reranker_driver()
        reranked = reranker.rerank(question, fused)
        return reranked[:top_k]


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
) -> list[dict]:
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
            entirely when given. Mainly for callers (the eval script) that
            call this repeatedly for the same question across strategies —
            without this, each call creates its own
            :class:`drivers.embedding.EmbeddingDriver` instance and
            re-triggers its lazy model load, which is real, avoidable cost
            when done many times in a loop. Production callers should
            leave this as ``None``.

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
    store = VectorStore()
    store.assert_dimension_matches(embedding_driver.dimension)

    if query_vector is None:
        logger.info("[query] Embedding question ...")
        query_vector = embedding_driver.embed_text(question)

    logger.info("[query] Vector-searching a candidate pool of %d chunks ...", candidate_k)
    vector_results = store.search(query_vector, top_k=candidate_k, min_score=0.0)

    if not _passes_relevance_gate(vector_results, top_k=k, min_score=threshold):
        logger.info("[query] No relevant chunks found.")
        return []

    active_strategy = strategy if strategy is not None else get_retrieval_strategy()
    logger.info("[query] Selecting final chunks via %s ...", type(active_strategy).__name__)
    chunks = active_strategy.select_chunks(
        question, vector_results, store, top_k=k, min_score=threshold
    )

    scores_str = ", ".join(f"{c['score']:.4f}" for c in chunks)
    logger.info("[query] Using %d chunk(s) as context. Scores: %s", len(chunks), scores_str)
    return chunks


def query_knowledge_base(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
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

    Returns:
        A string answer grounded in the retrieved chunks, or
        :data:`NO_RESULTS_MESSAGE` if no relevant chunks were found.

    Raises:
        RuntimeError: If the active embedding driver's dimension doesn't
            match the existing document_chunks.embedding column.
    """
    chunks = retrieve_chunks(question, top_k=top_k, min_score=min_score)

    if not chunks:
        logger.info("[query] Returning fallback message.")
        return NO_RESULTS_MESSAGE

    logger.info("[query] Generating answer with LLM driver='%s' ...", settings.LLM_DRIVER)
    answer_driver = get_answer_driver()
    answer = answer_driver.answer(question=question, context_chunks=chunks)

    logger.info("[query] Done.")
    return answer


def _passes_relevance_gate(
    vector_results: list[dict], top_k: int, min_score: float
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
    return any(c["score"] >= min_score for c in vector_results[:top_k])
