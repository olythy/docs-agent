"""Reranker driver abstractions.

Defines the common interface (``RerankerDriver``) for the optional second
retrieval stage: given a wider, cheap candidate list (from hybrid search),
a cross-encoder scores each (question, chunk) pair *together* — more
accurate than an embedding's independent similarity, but too expensive to
run over the whole corpus, so it only ever reranks an already-small
candidate list.

The active driver is selected at runtime via ``settings.RERANKER_DRIVER``:
    - ``"none"``          → :class:`NoopRerankerDriver` (default, backward
                             compatible: candidates keep their hybrid-search
                             order, nothing extra is computed)
    - ``"cross_encoder"``  → :class:`CrossEncoderRerankerDriver`

Usage::

    from drivers.reranker import get_reranker_driver
    reranker = get_reranker_driver()
    reranked = reranker.rerank(question, candidate_chunks)
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import cache

from config import settings
from models import RetrievedChunk


class RerankerDriver(ABC):
    """Abstract base class for all reranker backends."""

    @abstractmethod
    def rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Reorder ``chunks`` by relevance to ``question``.

        Args:
            question: The user's query text.
            chunks: Candidate chunks (as returned by
                :func:`query.hybrid.reciprocal_rank_fusion`).

        Returns:
            The same chunks, in descending relevance order. Implementations
            may replace each chunk's ``score`` with their own — callers
            should treat the *order*, not the score's scale, as the
            reliable signal (consistent with the hybrid-search score's
            already-fused, non-comparable-across-stages nature).
        """


class NoopRerankerDriver(RerankerDriver):
    """Default driver: passes candidates through unchanged.

    Keeps current behavior (no reranking) as the default so that nobody's
    existing setup changes just from this feature existing.
    """

    def rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return ``chunks`` unchanged, in their existing order."""
        return chunks


class CrossEncoderRerankerDriver(RerankerDriver):
    """Reranker using a local cross-encoder model via sentence-transformers.

    Defaults to ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, a
    multilingual cross-encoder (trained on MMARCO, MS MARCO machine-
    translated into 14 languages) — chosen over the far more common
    English-only ``cross-encoder/ms-marco-MiniLM-L-6-v2`` because this
    project's content and embedding model are both multilingual
    (Hungarian + English); an English-only reranker would be a regression
    on Hungarian content specifically.

    The model is loaded lazily on first use, matching
    :class:`drivers.embedding.LocalSentenceTransformerDriver`'s pattern.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialise the driver without loading the model yet.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                ``settings.RERANKER_MODEL``.
        """
        self._model_name = model_name or settings.RERANKER_MODEL
        self._model = None  # Loaded lazily on first rerank call

    def _get_model(self):
        """Load and cache the CrossEncoder model.

        Returns:
            The loaded ``CrossEncoder`` instance.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Score every (question, chunk.content) pair and sort descending.

        Args:
            question: The user's query text.
            chunks: Candidate chunks.

        Returns:
            The same chunks, each with ``score`` replaced by the
            cross-encoder's relevance score, sorted descending. Returns
            ``[]`` unchanged for an empty candidate list, avoiding a
            pointless model load.
        """
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(question, chunk.content) for chunk in chunks]
        scores = model.predict(pairs)

        reranked = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked


@cache
def get_reranker_driver(driver_name: str | None = None) -> RerankerDriver:
    """Factory function: return the active reranker driver.

    Reads ``settings.RERANKER_DRIVER`` unless ``driver_name`` is given, which
    lets a caller (e.g. ``scripts/eval_cli.py``, comparing rerankers within
    one process) select a driver explicitly instead of mutating the global
    ``settings`` — which can't work anyway, since ``Settings`` is a frozen
    dataclass (confirmed live: assigning ``settings.RERANKER_DRIVER``
    directly raises ``dataclasses.FrozenInstanceError``).

    Cached with ``@cache`` so repeated calls with the same
    ``driver_name`` reuse the same instance and its loaded in-memory model
    instead of reloading from disk; ``maxsize=None`` rather than ``1``
    since a caller may legitimately want more than one driver name cached
    at once within a single process (e.g. an eval run comparing them).

    Args:
        driver_name: Reranker driver to instantiate ('none' or
            'cross_encoder'). Defaults to ``settings.RERANKER_DRIVER``.

    Returns:
        A :class:`RerankerDriver` instance ready to call.

    Raises:
        ValueError: If the resolved driver name is unknown.
    """
    driver_name = (driver_name or settings.RERANKER_DRIVER).lower()

    if driver_name == "none":
        return NoopRerankerDriver()
    if driver_name == "cross_encoder":
        return CrossEncoderRerankerDriver()

    raise ValueError(
        f"Unknown RERANKER_DRIVER: '{driver_name}'. "
        "Valid options are: 'none', 'cross_encoder'."
    )
