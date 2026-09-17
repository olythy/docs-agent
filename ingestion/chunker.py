"""Text chunking module.

Splits extracted page-level text into smaller, overlapping chunks suitable
for embedding and vector storage. Uses a simple word-count approximation
instead of a full tokenizer to avoid extra dependencies in a learning project.

A commonly cited rule of thumb: 1 token ≈ 0.75 words on average, i.e. ~1.33
tokens per word — subword tokenizers usually produce *more* tokens than
words, not fewer. So CHUNK_SIZE=500 words is roughly 500 / 0.75 ≈ 667
tokens, not 500. This 0.75 ratio (``settings.WORDS_PER_TOKEN``) is only an
English average — it varies per text, and morphologically rich languages
like Hungarian typically tokenize *worse* (fewer words per token) than this,
since subword tokenizer vocabularies are shared across many languages and
long inflected/compound words don't match existing vocab pieces as cleanly.
Lower ``WORDS_PER_TOKEN`` for that kind of content.

There is also no single "typical" token limit across sentence-transformer
models to size against: the actual default local model
(paraphrase-multilingual-MiniLM-L12-v2) truncates at only 128 tokens,
confirmed empirically. Text beyond a model's limit is truncated silently
during embedding, not rejected — see :func:`validate_chunk_size_against_model`,
which warns when the configured chunk size likely exceeds the active
driver's limit.

Key exports:
    chunk_pages  -- Convert a list of page dicts into a flat list of chunk dicts.
    validate_chunk_size_against_model  -- Warn if chunks likely get truncated.
    get_chunk_overflow_strategy  -- Factory for the active CHUNK_OVERFLOW_STRATEGY.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

from config import settings
from drivers.embedding import EmbeddingDriver


def _split_words_into_chunks(
    words: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[str]]:
    """Slide a window of ``chunk_size`` words over ``words`` with ``chunk_overlap`` overlap.

    Args:
        words: The full list of words to split.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of words shared between adjacent chunks.

    Returns:
        A list of word-lists, each representing one chunk.

    Raises:
        ValueError: If ``chunk_size`` is not positive, ``chunk_overlap`` is
            negative, or ``chunk_overlap >= chunk_size``. The last case would
            make ``step`` zero or negative below, so the sliding window would
            never advance past ``len(words)`` and this function would loop
            forever.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
            f"({chunk_size}) — otherwise the sliding window never advances "
            "and this function loops forever."
        )

    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(chunk_words)
        start += step

    return chunks


def validate_chunk_size_against_model(
    chunk_size: int,
    max_seq_length: int | None,
    words_per_token: float | None = None,
) -> None:
    """Warn if ``chunk_size`` (in words) likely exceeds the model's token limit.

    Uses a words-per-token approximation (see module docstring) — purely
    advisory, does not raise, since exceeding the limit degrades retrieval
    quality rather than crashing anything, and some drivers (e.g. OpenAI)
    have no meaningful limit at realistic chunk sizes.

    Args:
        chunk_size: The configured chunk size, in words.
        max_seq_length: The active embedding driver's max sequence length in
            tokens, or ``None`` if the driver has no practical limit to check
            (see :meth:`drivers.embedding.EmbeddingDriver.max_sequence_length`).
        words_per_token: Override for ``settings.WORDS_PER_TOKEN``.

    Raises:
        ValueError: If the resolved ``words_per_token`` is not positive.
    """
    if max_seq_length is None:
        return

    ratio = words_per_token if words_per_token is not None else settings.WORDS_PER_TOKEN
    if ratio <= 0:
        raise ValueError(f"words_per_token must be positive, got {ratio}.")

    estimated_tokens = round(chunk_size / ratio)
    if estimated_tokens > max_seq_length:
        warnings.warn(
            f"CHUNK_SIZE={chunk_size} words (~{estimated_tokens} estimated "
            f"tokens) likely exceeds this model's max_seq_length="
            f"{max_seq_length} tokens. Chunks will be silently truncated "
            "during embedding — the truncated tail becomes invisible to "
            "retrieval. Lower CHUNK_SIZE or choose a different model.",
            stacklevel=2,
        )


def chunk_pages(
    pages: list[dict],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split page-level text into overlapping word-based chunks.

    Each chunk dict is ready to be inserted into the ``document_chunks`` table:
    it carries the raw text content and a ``metadata`` dict that records where
    in the source document the chunk came from.

    Args:
        pages: Page dicts as returned by :func:`ingestion.pdf_loader.extract_pages`.
            Each dict must have ``page_number`` (int) and ``text`` (str).
        source_file: The basename of the source PDF (e.g. ``"invoice.pdf"``).
            Stored in metadata so answers can cite their source.
        chunk_size: Override for ``settings.CHUNK_SIZE`` (words per chunk).
        chunk_overlap: Override for ``settings.CHUNK_OVERLAP`` (overlap in words).

    Returns:
        A flat list of chunk dicts, each containing:
            - ``content`` (str): The chunk text, ready for embedding.
            - ``metadata`` (dict): ``source_file``, ``page_number``,
              ``chunk_index`` (0-based, global across the whole document).

    Raises:
        ValueError: If the resolved ``chunk_overlap >= chunk_size`` (see
            :func:`_split_words_into_chunks`).
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    chunks: list[dict] = []
    global_chunk_index = 0

    for page in pages:
        text = page["text"].strip()
        if not text:
            # Skip blank/scanned pages — no content to chunk
            continue

        words = text.split()
        word_groups = _split_words_into_chunks(words, size, overlap)

        for word_group in word_groups:
            content = " ".join(word_group)
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "chunk_index": global_chunk_index,
                    },
                }
            )
            global_chunk_index += 1

    return chunks


def _split_oversized_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_seq_length: int,
) -> list[str]:
    """Split ``text`` into pieces that each fit within ``max_seq_length`` real tokens.

    Binary-searches, per piece, for the longest word-prefix whose real token
    count (via ``count_tokens``) still fits, then recurses on the remainder.
    Unlike the word-count chunking above, this checks ground truth after
    every guess instead of estimating — it's only affordable here because
    it only runs on the rare chunk that actually overflows.

    Args:
        text: The oversized chunk's text.
        count_tokens: Returns the real token count for a given string
            (e.g. :meth:`drivers.embedding.EmbeddingDriver.count_tokens`).
        max_seq_length: The token budget each returned piece must fit within.

    Returns:
        One or more pieces whose concatenation (with single spaces) equals
        ``text``. A single word longer than ``max_seq_length`` tokens on its
        own is returned as its own (still-oversized) piece — word-level
        granularity can't split it any finer.
    """
    words = text.split()
    if not words:
        return []
    if count_tokens(text) <= max_seq_length:
        return [text]

    lo, hi, fit = 1, len(words), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[:mid])) <= max_seq_length:
            fit = mid
            lo = mid + 1
        else:
            hi = mid - 1

    head = " ".join(words[:fit])
    rest = words[fit:]
    if not rest:
        return [head]
    return [head, *_split_oversized_text(" ".join(rest), count_tokens, max_seq_length)]


class ChunkOverflowStrategy(ABC):
    """Strategy for handling chunks that may exceed the embedding model's token limit.

    Selected at runtime via ``settings.CHUNK_OVERFLOW_STRATEGY`` (see
    :func:`get_chunk_overflow_strategy`) — same Strategy/Driver pattern as
    :class:`drivers.embedding.EmbeddingDriver`.
    """

    @abstractmethod
    def apply(self, chunks: list[dict], driver: EmbeddingDriver) -> list[dict]:
        """Return the (possibly modified) list of chunks to actually store.

        Args:
            chunks: Chunk dicts as produced by :func:`chunk_pages`.
            driver: The active embedding driver, queried for its token limit
                (and, for strategies that need it, real token counts).
        """


class WarnOverflowStrategy(ChunkOverflowStrategy):
    """Default, backward-compatible behavior: estimate and warn, don't correct.

    Chunks that overflow are still stored and silently truncated at embed
    time — this strategy only makes that risk visible via a log warning.
    """

    def apply(self, chunks: list[dict], driver: EmbeddingDriver) -> list[dict]:
        validate_chunk_size_against_model(settings.CHUNK_SIZE, driver.max_sequence_length())
        return chunks


class SplitOverflowStrategy(ChunkOverflowStrategy):
    """Corrective strategy: re-split any chunk that actually overflows.

    Uses the driver's real tokenizer (:meth:`EmbeddingDriver.count_tokens`)
    rather than the ``WORDS_PER_TOKEN`` estimate, so nothing is ever
    silently truncated — at the cost of an extra tokenizer call per chunk.
    Falls back to :class:`WarnOverflowStrategy` when the active driver can't
    report real token counts (e.g. ``OpenAIEmbeddingDriver``): there's no
    ground truth to split against, so correction isn't possible, only the
    same estimate-based warning is.
    """

    def apply(self, chunks: list[dict], driver: EmbeddingDriver) -> list[dict]:
        max_seq_length = driver.max_sequence_length()
        if max_seq_length is None:
            return chunks

        if chunks and driver.count_tokens(chunks[0]["content"]) is None:
            warnings.warn(
                "CHUNK_OVERFLOW_STRATEGY=split requires the active embedding "
                f"driver ({type(driver).__name__}) to support real token "
                "counting, which it doesn't. Falling back to the "
                "WORDS_PER_TOKEN-estimate warning instead.",
                stacklevel=2,
            )
            return WarnOverflowStrategy().apply(chunks, driver)

        corrected: list[dict] = []
        next_index = 0
        for chunk in chunks:
            for piece in _split_oversized_text(
                chunk["content"], driver.count_tokens, max_seq_length
            ):
                corrected.append(
                    {
                        "content": piece,
                        "metadata": {**chunk["metadata"], "chunk_index": next_index},
                    }
                )
                next_index += 1
        return corrected


def get_chunk_overflow_strategy() -> ChunkOverflowStrategy:
    """Factory function: return the active strategy from ``settings.CHUNK_OVERFLOW_STRATEGY``.

    Raises:
        ValueError: If ``CHUNK_OVERFLOW_STRATEGY`` is set to an unknown value.
    """
    name = settings.CHUNK_OVERFLOW_STRATEGY.lower()

    if name == "warn":
        return WarnOverflowStrategy()
    if name == "split":
        return SplitOverflowStrategy()

    raise ValueError(
        f"Unknown CHUNK_OVERFLOW_STRATEGY: '{name}'. Valid options are: 'warn', 'split'."
    )
