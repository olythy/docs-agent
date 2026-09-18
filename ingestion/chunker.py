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
from typing import ClassVar

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


def _largest_fitting_prefix(
    words: list[str],
    count_tokens: Callable[[str], int],
    budget: int,
) -> int:
    """Binary-search the largest word-prefix of ``words`` whose real token count fits ``budget``.

    Always returns at least 1, even if the first word alone exceeds
    ``budget`` — word-level granularity can't split a single word any
    finer, so it's returned oversized rather than dropped or split further.
    """
    lo, hi, fit = 1, len(words), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[:mid])) <= budget:
            fit = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return fit


def _split_oversized_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_seq_length: int,
) -> list[str]:
    """Split ``text`` into balanced pieces that each fit within ``max_seq_length`` real tokens.

    Greedily maxing out each piece up to the hard token limit sounds
    optimal but isn't: for a chunk only slightly over the limit (say 132
    tokens vs. a 128 limit), it produces one full 128-token piece and a
    near-empty ~4-token straggler. That straggler is nearly useless as its
    own embedding — too little semantic content for retrieval to ever
    match it well. Instead, this first estimates how many pieces are
    actually needed (``ceil(total_tokens / max_seq_length)``) and aims
    each piece at an even share of that, so e.g. 132 tokens over a 128
    limit becomes two ~66-token pieces instead of 128 + 4.

    Still checks ground truth via ``count_tokens`` after every guess
    (binary search), same as a pure max-out approach would — the
    balancing only changes the *target* each piece aims for, not the
    correctness guarantee that no returned piece exceeds ``max_seq_length``.

    Args:
        text: The oversized chunk's text.
        count_tokens: Returns the real token count for a given string
            (e.g. :meth:`drivers.embedding.EmbeddingDriver.count_tokens`).
        max_seq_length: The token budget each returned piece must fit within.

    Returns:
        One or more pieces whose concatenation (with single spaces) equals
        ``text``. A single word longer than ``max_seq_length`` tokens on its
        own is returned as its own (still-oversized) piece.
    """
    words = text.split()
    if not words:
        return []
    total_tokens = count_tokens(text)
    if total_tokens <= max_seq_length:
        return [text]

    pieces_left = -(-total_tokens // max_seq_length)  # ceil division
    pieces: list[str] = []
    remaining = words

    while remaining and pieces_left > 0:
        remaining_tokens = count_tokens(" ".join(remaining))
        target = -(-remaining_tokens // pieces_left)  # ceil division
        budget = min(target, max_seq_length)

        fit = _largest_fitting_prefix(remaining, count_tokens, budget)
        pieces.append(" ".join(remaining[:fit]))
        remaining = remaining[fit:]
        pieces_left -= 1

    if remaining:
        # Token density wasn't uniform enough for the balanced estimate to
        # fully consume the text in the predicted number of pieces (rare).
        # Finish correctly via the same algorithm, just less evenly.
        pieces.extend(_split_oversized_text(" ".join(remaining), count_tokens, max_seq_length))

    return pieces


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

        if not driver.supports_token_counting():
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


class ChunkingStrategy(ABC):
    """Strategy for turning a whole document's text into chunks.

    A different responsibility from :class:`ChunkOverflowStrategy` — this is
    about *how chunks are produced in the first place*, not what happens
    when one turns out to be too big. Selected at runtime via
    ``settings.CHUNKING_STRATEGY`` (see :func:`get_chunking_strategy`), same
    Strategy/Driver pattern as :class:`drivers.embedding.EmbeddingDriver`.
    """

    @abstractmethod
    def split(self, full_text: str, driver: EmbeddingDriver) -> list[tuple[str, int]]:
        """Return (chunk_text, start_word_index) pairs, in document order.

        ``start_word_index`` is this chunk's position in ``full_text.split()``
        — :func:`chunk_document` uses it to look up which page(s) the chunk's
        words came from. It must come from the strategy's own splitting
        logic, not from searching for ``chunk_text`` inside ``full_text``:
        overlapping chunks (e.g. a repeated word at the seam) make naive
        substring search find the wrong occurrence.

        Args:
            full_text: The whole document's text, as produced by
                :func:`ingestion.pdf_loader.extract_document_text`.
            driver: The active embedding driver — strategies that size
                chunks in tokens need :meth:`EmbeddingDriver.max_sequence_length`
                and :meth:`EmbeddingDriver.count_tokens`.
        """


class WordChunkingStrategy(ChunkingStrategy):
    """The original word-count sliding window, operating on the whole document.

    Reuses :func:`_split_words_into_chunks` unchanged — same windowing math
    as :func:`chunk_pages`, just applied to the whole document's words
    instead of one page's at a time (which is what actually fixes the
    page-boundary truncation problem: there's no page loop here at all).
    """

    def split(self, full_text: str, driver: EmbeddingDriver) -> list[tuple[str, int]]:
        words = full_text.split()
        groups = _split_words_into_chunks(words, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
        return [(" ".join(group), i * step) for i, group in enumerate(groups)]


class LangChainChunkingStrategy(ChunkingStrategy):
    """Paragraph-/sentence-aware chunking via ``langchain_text_splitters``.

    Tries each separator in turn (paragraph, line, sentence, clause, word)
    and only falls back to a cruder one when a piece still doesn't fit —
    unlike the word strategy's blind fixed-size window, this keeps whole
    paragraphs/sentences together whenever they fit the budget.

    Uses ``chunk_overlap=0`` deliberately: paragraph/sentence-aware splits
    already preserve more context per chunk than blind word-overlap did,
    and non-overlapping pieces are what makes mapping each piece back to a
    word-index range unambiguous (see :meth:`split`'s docstring on
    :class:`ChunkingStrategy` for why that matters).

    Also uses ``keep_separator=False``: confirmed empirically that the
    default (``True``) leaves a piece starting with a bare leftover
    separator when a split happens at ``". "``/``", "`` (e.g. a piece
    reading ``". word6 word7"`` instead of ``"word6 word7"``) — ugly in
    the embedded text, and the reason :meth:`split` below needs
    character-offset tracking instead of plain word-counting.
    ``keep_separator=False`` was confirmed (same empirical check) to make
    every split land cleanly on a word boundary for every separator in
    ``_SEPARATORS`` except the last, empty-string fallback, which only
    kicks in for a single "word" longer than the whole chunk budget — the
    same rare, accepted edge case :func:`_split_oversized_text` documents
    for the word strategy. Character-offset tracking (below) stays in
    place as the correct, general answer either way — this isn't a
    "fixed it, remove the safety net" situation, it just shrinks how
    often that net is the thing doing the work.
    """

    _SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", ", ", " ", ""]

    def split(self, full_text: str, driver: EmbeddingDriver) -> list[tuple[str, int]]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if driver.supports_token_counting():
            length_function = driver.count_tokens
        else:
            length_function = len
            warnings.warn(
                f"CHUNKING_STRATEGY=langchain with {type(driver).__name__}, which has no "
                "real tokenizer: chunk_size is measured in raw characters instead of "
                "tokens, and falls back to settings.CHUNK_SIZE if the driver also has no "
                "max_sequence_length() — CHUNK_SIZE is documented in *words*, so chunks "
                "will likely come out far smaller than intended.",
                stacklevel=2,
            )
        chunk_size = driver.max_sequence_length() or settings.CHUNK_SIZE
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=length_function,
            separators=self._SEPARATORS,
            keep_separator=False,
        )
        pieces = splitter.split_text(full_text)

        # Pieces are non-overlapping substrings of full_text (chunk_overlap=0)
        # that, with keep_separator=False, land on a word boundary as
        # .split() sees it in every observed case except the rare
        # single-oversized-word fallback (see class docstring) — so
        # word-count alone still can't be trusted in general. Character
        # offsets are exact; converting the offset to a word-index via
        # counting words in the text *before* it is exact too, since the
        # split point is on whitespace whenever that assumption holds.
        results = []
        char_cursor = 0
        for piece in pieces:
            char_start = full_text.find(piece, char_cursor)
            word_start = len(full_text[:char_start].split())
            results.append((piece, word_start))
            char_cursor = char_start + len(piece)
        return results


def get_chunking_strategy() -> ChunkingStrategy:
    """Factory function: return the active strategy from ``settings.CHUNKING_STRATEGY``.

    Raises:
        ValueError: If ``CHUNKING_STRATEGY`` is set to an unknown value.
    """
    name = settings.CHUNKING_STRATEGY.lower()

    if name == "word":
        return WordChunkingStrategy()
    if name == "langchain":
        return LangChainChunkingStrategy()

    raise ValueError(
        f"Unknown CHUNKING_STRATEGY: '{name}'. Valid options are: 'word', 'langchain'."
    )


def chunk_document(
    full_text: str,
    word_page_map: list[int],
    source_file: str,
    driver: EmbeddingDriver,
) -> list[dict]:
    """Split a whole document's text into chunk dicts, using the active CHUNKING_STRATEGY.

    The document-level counterpart to :func:`chunk_pages` — same chunk-dict
    shape (``content``, ``metadata: {source_file, page_number, chunk_index}``),
    but chunked from :func:`ingestion.pdf_loader.extract_document_text`'s
    output instead of per-page text, which is what avoids splitting a
    paragraph that spans a page break into two truncated chunks.

    ``page_number`` is assigned by majority vote: whichever page contributed
    the most words to a chunk. A chunk's words very rarely straddle more
    than two pages, and an occasional off-by-one-page citation is a
    negligible cost for not having to change the metadata shape (a page
    *range* would ripple into the prompt template and retrieval display).

    Args:
        full_text: The whole document's text.
        word_page_map: Page number per word in ``full_text.split()`` (same
            length), as returned by ``extract_document_text``.
        source_file: The basename of the source PDF, stored in metadata.
        driver: The active embedding driver, passed through to the strategy.

    Returns:
        A flat list of chunk dicts, in document order.
    """
    from collections import Counter

    chunks = []
    for i, (content, start_word) in enumerate(get_chunking_strategy().split(full_text, driver)):
        n_words = len(content.split())
        pages_in_chunk = word_page_map[start_word : start_word + n_words]
        page_number = Counter(pages_in_chunk).most_common(1)[0][0] if pages_in_chunk else None
        chunks.append(
            {
                "content": content,
                "metadata": {
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_index": i,
                },
            }
        )
    return chunks
