"""Tests for ingestion.chunker: word-based chunking with overlap."""

import warnings
from unittest.mock import MagicMock

import pytest

import ingestion.chunker as chunker_module
from ingestion.chunker import (
    LangChainChunkingStrategy,
    SplitOverflowStrategy,
    WarnOverflowStrategy,
    WordChunkingStrategy,
    _find_natural_break_point,
    _package_chunk,
    _split_oversized_text,
    _split_words_into_chunks,
    _strip_header_prefix,
    chunk_document,
    chunk_pages,
    enrich_chunk_content,
    get_chunk_overflow_strategy,
    get_chunking_strategy,
    validate_chunk_size_against_model,
)
from models import Chunk, ChunkMetadata


def _fake_driver(max_seq_length=None, supports_tokens=False):
    driver = MagicMock()
    driver.max_sequence_length.return_value = max_seq_length
    driver.supports_token_counting.return_value = supports_tokens
    driver.count_tokens.side_effect = lambda t: len(t.split())
    return driver


def test_split_empty_words_returns_empty_list():
    assert _split_words_into_chunks([], chunk_size=5, chunk_overlap=1) == []


def test_split_no_overlap_splits_into_equal_groups():
    words = ["a", "b", "c", "d", "e", "f"]
    result = _split_words_into_chunks(words, chunk_size=2, chunk_overlap=0)
    assert result == [["a", "b"], ["c", "d"], ["e", "f"]]


def test_split_last_group_may_be_shorter():
    words = ["a", "b", "c", "d", "e"]
    result = _split_words_into_chunks(words, chunk_size=2, chunk_overlap=0)
    assert result == [["a", "b"], ["c", "d"], ["e"]]


def test_split_overlap_repeats_words_between_adjacent_groups():
    words = ["a", "b", "c", "d"]
    result = _split_words_into_chunks(words, chunk_size=3, chunk_overlap=1)
    # step = chunk_size - chunk_overlap = 2: [0:3], [2:4] — "c" appears in both.
    assert result == [["a", "b", "c"], ["c", "d"]]


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(5, 5), (5, 6)])
def test_split_raises_when_overlap_not_smaller_than_size(chunk_size, chunk_overlap):
    """The exact bug from the code review: overlap >= size used to hang forever."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        _split_words_into_chunks(["a"], chunk_size, chunk_overlap)


def test_split_raises_on_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        _split_words_into_chunks(["a"], chunk_size=0, chunk_overlap=0)


def test_split_raises_on_negative_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        _split_words_into_chunks(["a"], chunk_size=5, chunk_overlap=-1)


def test_chunk_pages_builds_content_and_metadata():
    pages = [{"page_number": 1, "text": "one two three four"}]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=2, chunk_overlap=0)

    assert [c.content for c in chunks] == ["one two", "three four"]
    assert chunks[0].metadata == ChunkMetadata(
        source_file="doc.pdf", page_number=1, chunk_index=0
    )
    assert chunks[1].metadata.chunk_index == 1


def test_chunk_pages_skips_blank_pages():
    pages = [
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": "hello world"},
    ]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=5, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].metadata.page_number == 2


def test_chunk_pages_chunk_index_is_global_across_pages():
    pages = [
        {"page_number": 1, "text": "a b"},
        {"page_number": 2, "text": "c d"},
    ]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=2, chunk_overlap=0)
    assert [c.metadata.chunk_index for c in chunks] == [0, 1]


def test_chunk_pages_uses_settings_defaults_when_not_overridden(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNK_SIZE=3, CHUNK_OVERLAP=0),
    )
    pages = [{"page_number": 1, "text": "one two three four five six"}]

    chunks = chunk_pages(pages, source_file="doc.pdf")

    assert [c.content for c in chunks] == ["one two three", "four five six"]


def test_chunk_pages_propagates_invalid_overlap_error():
    pages = [{"page_number": 1, "text": "one two three"}]
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages(pages, source_file="doc.pdf", chunk_size=3, chunk_overlap=3)


def test_validate_chunk_size_no_warning_when_max_seq_length_unknown():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_chunk_size_against_model(chunk_size=99999, max_seq_length=None)


def test_validate_chunk_size_no_warning_when_within_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_chunk_size_against_model(chunk_size=50, max_seq_length=128)


def test_validate_chunk_size_warns_when_exceeding_limit():
    with pytest.warns(UserWarning, match="CHUNK_SIZE"):
        validate_chunk_size_against_model(chunk_size=100, max_seq_length=128)


def test_validate_chunk_size_warns_for_the_actual_default_config():
    """Regression test for the real bug this guard exists for.

    The default CHUNK_SIZE=500 vs. the default local model's real
    max_seq_length=128 — confirmed empirically:
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').max_seq_length
    is 128, not the 512 the old chunker.py docstring assumed.
    """
    with pytest.warns(UserWarning, match="CHUNK_SIZE=500"):
        validate_chunk_size_against_model(chunk_size=500, max_seq_length=128)


def test_validate_chunk_size_uses_settings_default_ratio(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(WORDS_PER_TOKEN=0.75)
    )
    # 100 words / 0.75 ~= 133 tokens > 128 -> warns
    with pytest.warns(UserWarning, match="CHUNK_SIZE"):
        validate_chunk_size_against_model(chunk_size=100, max_seq_length=128)


def test_validate_chunk_size_explicit_ratio_overrides_settings(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(WORDS_PER_TOKEN=0.75)
    )
    # A worse (lower) ratio makes the same chunk_size look larger in tokens.
    with pytest.warns(UserWarning, match="CHUNK_SIZE"):
        validate_chunk_size_against_model(
            chunk_size=50, max_seq_length=128, words_per_token=0.3
        )


def test_validate_chunk_size_raises_on_non_positive_ratio():
    with pytest.raises(ValueError, match="words_per_token"):
        validate_chunk_size_against_model(
            chunk_size=50, max_seq_length=128, words_per_token=0
        )


# --- _split_oversized_text ---


def _word_count(text: str) -> int:
    """Fake count_tokens: 1 word = 1 token, used throughout this section."""
    return len(text.split())


def test_split_oversized_text_empty_returns_empty_list():
    assert _split_oversized_text("", _word_count, max_seq_length=5) == []


def test_split_oversized_text_returns_unchanged_when_within_budget():
    assert _split_oversized_text("a b c", _word_count, max_seq_length=5) == ["a b c"]


def test_split_oversized_text_splits_into_pieces_that_fit():
    text = " ".join(f"w{i}" for i in range(10))
    pieces = _split_oversized_text(text, _word_count, max_seq_length=3)
    assert all(_word_count(p) <= 3 for p in pieces)
    assert " ".join(pieces) == text  # nothing lost, order preserved


def test_split_oversized_text_single_word_over_budget_returned_as_is():
    """Word-level granularity can't split a single oversized word any finer."""
    result = _split_oversized_text(
        "supercalifragilisticexpialidocious", lambda t: 100, max_seq_length=5
    )
    assert result == ["supercalifragilisticexpialidocious"]


def test_split_oversized_text_balances_pieces_instead_of_a_tiny_straggler():
    """Regression test for a real finding: greedily maxing out each piece up to
    max_seq_length left a near-empty trailing piece when a chunk was only
    slightly over the limit (a real document produced 128 + 6 tokens from
    a 50-word/132-token chunk). Aiming each piece at an even share of the
    total instead should produce two comparably-sized pieces.
    """
    words = [f"w{i}" for i in range(50)]
    # Position-aware weights, like a real tokenizer's uneven density:
    # the first 40 words cost 3 tokens each, the last 10 cost 1 each.
    weights = {f"w{i}": (3 if i < 40 else 1) for i in range(50)}

    def count_tokens(text: str) -> int:
        return sum(weights[w] for w in text.split())

    pieces = _split_oversized_text(" ".join(words), count_tokens, max_seq_length=90)

    token_counts = [count_tokens(p) for p in pieces]
    assert all(t <= 90 for t in token_counts)
    assert len(pieces) == 2
    # The old greedy-max algorithm would produce 90 + 40 here (maxing the
    # first piece to the hard limit); balancing keeps both pieces close.
    assert min(token_counts) >= 0.9 * max(token_counts)


def test_split_oversized_text_with_overlap():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    # fit=10, overlap_ratio=0.2 -> overlap_count=2 words
    pieces = _split_oversized_text(
        text, _word_count, max_seq_length=10, overlap_ratio=0.2
    )
    assert len(pieces) >= 2
    first_words = pieces[0].split()
    second_words = pieces[1].split()
    assert first_words[-2:] == second_words[:2]


def test_find_natural_break_point_finds_sentence_terminators():
    # Last word has a period -> break point includes that word
    words = ["Sentence", "one.", "Sentence", "two", "starts", "here."]
    assert _find_natural_break_point(words) == 6

    # Terminator in the second half (index 2 out of 5 words = 40% < 50%, index 3 = 60% >= 50%)
    words_mid = ["word0", "word1", "word2", "word3.", "word4", "word5"]
    assert _find_natural_break_point(words_mid) == 4

    # Trailing quote after period handled
    words_quote = ["He", "said,", "yes.", '"Extra"']
    assert _find_natural_break_point(words_quote) == 3


def test_find_natural_break_point_ignores_terminators_below_threshold():
    # Period is at index 0 (< 50% of 6 words) -> fallback to all words (no tiny fragment)
    words = ["Start.", "word1", "word2", "word3", "word4", "word5"]
    assert _find_natural_break_point(words) == 6


def test_split_oversized_text_cuts_at_sentence_boundary():
    # Two clear sentences. Budget is enough for 7 words, but sentence 1 ends at word 3.
    text = "First phrase here. Second longer phrase goes here."
    pieces = _split_oversized_text(
        text, _word_count, max_seq_length=7, overlap_ratio=0.0
    )
    assert len(pieces) == 2
    assert pieces[0] == "First phrase here."
    assert pieces[1] == "Second longer phrase goes here."


# --- WarnOverflowStrategy ---


def _chunk(content: str, **metadata_kwargs) -> Chunk:
    """Build a Chunk for tests, filling in ChunkMetadata's required fields with sensible defaults."""
    metadata_kwargs.setdefault("source_file", "doc.pdf")
    metadata_kwargs.setdefault("page_number", None)
    metadata_kwargs.setdefault("chunk_index", 0)
    return Chunk(content=content, metadata=ChunkMetadata(**metadata_kwargs))


def test_warn_strategy_returns_chunks_unchanged():
    driver = MagicMock()
    driver.max_sequence_length.return_value = None
    chunks = [_chunk("a b c")]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = WarnOverflowStrategy().apply(chunks, driver)

    assert result is chunks


def test_warn_strategy_warns_via_validate_chunk_size(monkeypatch, settings_override):
    monkeypatch.setattr(chunker_module, "settings", settings_override(CHUNK_SIZE=500))
    driver = MagicMock()
    driver.max_sequence_length.return_value = 128
    chunks = [_chunk("x")]

    with pytest.warns(UserWarning, match="CHUNK_SIZE"):
        result = WarnOverflowStrategy().apply(chunks, driver)

    assert result == chunks


# --- SplitOverflowStrategy ---


def test_split_strategy_returns_unchanged_when_no_limit():
    driver = MagicMock()
    driver.max_sequence_length.return_value = None
    chunks = [_chunk("a b c")]

    assert SplitOverflowStrategy().apply(chunks, driver) is chunks


def test_split_strategy_with_empty_chunks_returns_empty():
    driver = MagicMock()
    driver.max_sequence_length.return_value = 128

    assert SplitOverflowStrategy().apply([], driver) == []


def test_split_strategy_splits_oversized_chunk_and_reindexes():
    driver = MagicMock()
    driver.max_sequence_length.return_value = 3
    driver.count_tokens.side_effect = _word_count
    chunks = [
        _chunk("a b c d e f", page_number=1, chunk_index=0),
        _chunk("g h", page_number=1, chunk_index=1),
    ]

    result = SplitOverflowStrategy().apply(chunks, driver)

    assert [c.content for c in result] == ["a b c", "d e f", "g h"]
    assert [c.metadata.chunk_index for c in result] == [0, 1, 2]
    assert all(c.metadata.source_file == "doc.pdf" for c in result)


def test_split_strategy_applies_overlap():
    driver = MagicMock()
    driver.max_sequence_length.return_value = 10
    driver.count_tokens.side_effect = _word_count
    words = [f"w{i}" for i in range(20)]
    chunks = [_chunk(" ".join(words))]

    strategy = SplitOverflowStrategy(overlap_ratio=0.2)
    result = strategy.apply(chunks, driver)

    assert len(result) >= 2
    first_words = result[0].content.split()
    second_words = result[1].content.split()
    assert first_words[-2:] == second_words[:2]


def test_split_strategy_falls_back_to_warn_when_driver_lacks_real_token_counts(
    monkeypatch, settings_override
):
    monkeypatch.setattr(chunker_module, "settings", settings_override(CHUNK_SIZE=500))
    driver = MagicMock()
    driver.max_sequence_length.return_value = 128
    driver.supports_token_counting.return_value = False
    chunks = [_chunk("x")]

    with pytest.warns(UserWarning) as record:
        result = SplitOverflowStrategy().apply(chunks, driver)

    assert result == chunks
    messages = [str(w.message) for w in record]
    assert any("Falling back" in m for m in messages)


def test_split_strategy_preserves_header_path_across_all_pieces():
    header = "# Main > ## Section"
    driver = MagicMock()
    driver.max_sequence_length.return_value = 10
    driver.count_tokens.side_effect = _word_count

    # Content has header prefixed + body
    body = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    content = f"{header}\n\n{body}"
    chunks = [_chunk(content, source_file="doc.md", header_path=header)]

    result = SplitOverflowStrategy(overlap_ratio=0.0).apply(chunks, driver)
    assert len(result) >= 2
    for chunk in result:
        assert chunk.content.startswith(f"{header}\n\n")
        assert chunk.metadata.header_path == header
        # Check token budget respected for the whole piece
        assert driver.count_tokens(chunk.content) <= 10


def test_split_strategy_uses_settings_overlap_ratio(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNK_SPLIT_OVERLAP_RATIO=0.25)
    )
    strategy = SplitOverflowStrategy()
    assert strategy.overlap_ratio == 0.25


# --- get_chunk_overflow_strategy ---


def test_get_chunk_overflow_strategy_returns_split_by_default():
    assert isinstance(get_chunk_overflow_strategy(), SplitOverflowStrategy)


def test_get_chunk_overflow_strategy_returns_warn(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNK_OVERFLOW_STRATEGY="warn")
    )
    assert isinstance(get_chunk_overflow_strategy(), WarnOverflowStrategy)


def test_get_chunk_overflow_strategy_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNK_OVERFLOW_STRATEGY="bogus")
    )
    with pytest.raises(ValueError, match="Unknown CHUNK_OVERFLOW_STRATEGY"):
        get_chunk_overflow_strategy()


# --- WordChunkingStrategy ---


def test_word_chunking_strategy_returns_indexed_chunks(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNK_SIZE=3, CHUNK_OVERLAP=1)
    )
    full_text = "a b c d e f g"

    result = WordChunkingStrategy().split(full_text, driver=MagicMock())

    assert result == [("a b c", 0), ("c d e", 2), ("e f g", 4), ("g", 6)]


def test_word_chunking_strategy_operates_on_the_whole_document_not_per_page():
    """The whole point of chunk_document: no page loop, so a chunk can span
    what used to be a page boundary — confirmed by CHUNK_SIZE spanning the
    "page1 page2" join point below with no truncation.
    """
    full_text = "end of page one start of page two"
    words = full_text.split()

    result = WordChunkingStrategy().split(full_text, driver=MagicMock())

    # With default CHUNK_SIZE/CHUNK_OVERLAP (500/50), everything fits in
    # a single chunk, proving the boundary in the middle isn't special.
    assert len(result) == 1
    assert result[0][0].split() == words


# --- LangChainChunkingStrategy ---


def test_langchain_chunking_strategy_splits_on_paragraph_marker(
    monkeypatch, settings_override
):
    monkeypatch.setattr(chunker_module, "settings", settings_override(CHUNK_SIZE=60))
    full_text = (
        "one two three four five six seven eight nine ten"
        "\n\neleven twelve thirteen fourteen fifteen"
    )
    driver = _fake_driver(max_seq_length=None, supports_tokens=False)

    with pytest.warns(UserWarning, match="no real tokenizer"):
        result = LangChainChunkingStrategy().split(full_text, driver)

    assert result == [
        ("one two three four five six seven eight nine ten", 0),
        ("eleven twelve thirteen fourteen fifteen", 10),
    ]


def test_langchain_chunking_strategy_warns_when_driver_lacks_token_counting():
    """Regression test for a real finding: falling back to len() means
    chunk_size ends up measured in characters, and — if the driver also has
    no max_sequence_length() — defaults to settings.CHUNK_SIZE, which is
    documented in *words*, not characters. Silently mismatching units would
    produce chunks far smaller than the user configured, with no signal
    that anything is off.
    """
    driver = _fake_driver(max_seq_length=None, supports_tokens=False)

    with pytest.warns(UserWarning, match="raw characters instead of tokens"):
        LangChainChunkingStrategy().split("some text", driver)


def test_langchain_chunking_strategy_no_warning_when_driver_supports_token_counting():
    driver = _fake_driver(max_seq_length=128, supports_tokens=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LangChainChunkingStrategy().split("some text", driver)


def test_langchain_chunking_strategy_uses_driver_token_length_when_supported():
    """When the driver supports real token counting, chunk_size is measured
    in tokens (via count_tokens), not characters — and max_sequence_length
    (not settings.CHUNK_SIZE) sets the budget.
    """
    driver = _fake_driver(max_seq_length=3, supports_tokens=True)
    full_text = "one two three\n\nfour five six"

    result = LangChainChunkingStrategy().split(full_text, driver)

    assert result == [("one two three", 0), ("four five six", 3)]
    driver.count_tokens.assert_called()


def test_langchain_chunking_strategy_no_bare_separator_leftover(
    monkeypatch, settings_override
):
    """Regression test for a real finding: without keep_separator=False, a
    piece can start with a bare "." left over from a ". " separator split
    (e.g. ". Sentence two" instead of "Sentence two") — confirmed
    empirically, and ugly in the embedded text on top of anything else.
    """
    monkeypatch.setattr(chunker_module, "settings", settings_override(CHUNK_SIZE=20))
    driver = _fake_driver(max_seq_length=None, supports_tokens=False)
    full_text = "Sentence one here. Sentence two follows. Sentence three ends."

    with pytest.warns(UserWarning, match="no real tokenizer"):
        result = LangChainChunkingStrategy().split(full_text, driver)

    assert len(result) > 1, (
        "text must actually have been split for this test to mean anything"
    )
    for text, _start in result:
        assert not text.startswith((".", ",")), (
            f"piece starts with a bare separator: {text!r}"
        )


def test_langchain_chunking_strategy_start_indices_are_correct_even_with_stray_punctuation(
    monkeypatch, settings_override
):
    """Character-offset tracking (not word-count tracking) must get every
    start index right even in the one case keep_separator=False can't
    prevent: a single "word" longer than the whole chunk budget, which
    forces the last-resort "" separator and can land mid-word (see the
    class docstring's note on this accepted edge case).
    """
    monkeypatch.setattr(chunker_module, "settings", settings_override(CHUNK_SIZE=5))
    driver = _fake_driver(max_seq_length=None, supports_tokens=False)
    full_text = "a " + ("x" * 50) + " b c d e"

    with pytest.warns(UserWarning, match="no real tokenizer"):
        result = LangChainChunkingStrategy().split(full_text, driver)

    words = full_text.split()
    for _text, start in result:
        # Whatever the piece's own tokenization looks like, the start index
        # must point at a real position within the original word list.
        assert 0 <= start <= len(words)
    # Pieces are produced in non-decreasing start order (chunk_overlap=0).
    starts = [start for _, start in result]
    assert starts == sorted(starts)


# --- get_chunking_strategy ---


def test_get_chunking_strategy_returns_word_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNKING_STRATEGY="word")
    )
    assert isinstance(get_chunking_strategy(), WordChunkingStrategy)


def test_get_chunking_strategy_returns_langchain(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNKING_STRATEGY="langchain")
    )
    assert isinstance(get_chunking_strategy(), LangChainChunkingStrategy)


def test_get_chunking_strategy_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module, "settings", settings_override(CHUNKING_STRATEGY="bogus")
    )
    with pytest.raises(ValueError, match="Unknown CHUNKING_STRATEGY"):
        get_chunking_strategy()


# --- chunk_document ---


def test_chunk_document_builds_metadata_from_strategy_output(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNKING_STRATEGY="word", CHUNK_SIZE=3, CHUNK_OVERLAP=0),
    )
    full_text = "a b c d e f"
    word_page_map = [1, 1, 1, 2, 2, 2]

    chunks = chunk_document(
        full_text, word_page_map, source_file="doc.pdf", driver=MagicMock()
    )

    assert [c.content for c in chunks] == ["a b c", "d e f"]
    assert [c.metadata.page_number for c in chunks] == [1, 2]
    assert [c.metadata.chunk_index for c in chunks] == [0, 1]
    assert all(c.metadata.source_file == "doc.pdf" for c in chunks)


def test_chunk_document_page_number_is_majority_vote_across_a_page_boundary(
    monkeypatch, settings_override
):
    """A chunk whose words straddle a page boundary gets the page that
    contributed the most words — the accepted tradeoff (see chunk_document's
    docstring) instead of a page range.
    """
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNKING_STRATEGY="word", CHUNK_SIZE=5, CHUNK_OVERLAP=0),
    )
    full_text = "a b c d e"
    word_page_map = [1, 1, 1, 2, 2]  # 3 words from page 1, 2 from page 2

    chunks = chunk_document(
        full_text, word_page_map, source_file="doc.pdf", driver=MagicMock()
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.page_number == 1


def test_chunk_document_uses_langchain_strategy_when_configured(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNKING_STRATEGY="langchain", CHUNK_SIZE=60),
    )
    full_text = "one two three four five six seven eight nine ten\n\neleven twelve"
    word_page_map = [1] * 10 + [2] * 2

    with pytest.warns(UserWarning, match="no real tokenizer"):
        chunks = chunk_document(
            full_text,
            word_page_map,
            source_file="doc.pdf",
            driver=_fake_driver(max_seq_length=None, supports_tokens=False),
        )

    assert [c.content for c in chunks] == [
        "one two three four five six seven eight nine ten",
        "eleven twelve",
    ]
    assert [c.metadata.page_number for c in chunks] == [1, 2]


def test_enrich_chunk_content():
    # 1. Empty header_path leaves content untouched
    assert enrich_chunk_content("hello world", "") == "hello world"

    # 2. Already starts with full header path
    path = "# Doc > ## Section"
    assert enrich_chunk_content(f"{path}\n\nhello", path) == f"{path}\n\nhello"

    # 3. Starts with leaf header line: replaces with full path
    leaf_chunk = "## Section\nSome details."
    enriched = enrich_chunk_content(leaf_chunk, path)
    assert enriched == "# Doc > ## Section\n\nSome details."

    # 4. Body text without header: prepends header
    body_chunk = "Some details without header."
    assert enrich_chunk_content(body_chunk, path) == f"{path}\n\n{body_chunk}"


class TestStripHeaderPrefix:
    """_strip_header_prefix is enrich_chunk_content's inverse — see its
    docstring for why the two are kept side by side."""

    PATH = "# Doc > ## Section"

    def test_empty_header_path_leaves_content_untouched(self):
        assert _strip_header_prefix("hello world", "") == "hello world"

    def test_strips_an_embedded_prefix(self):
        content = f"{self.PATH}\n\nhello world"
        assert _strip_header_prefix(content, self.PATH) == "hello world"

    def test_leaves_content_unchanged_when_prefix_not_present(self):
        assert _strip_header_prefix("hello world", self.PATH) == "hello world"

    def test_round_trips_with_enrich_chunk_content(self):
        """enrich() then strip() must return the original body — the whole
        point of keeping the two functions paired."""
        body = "Some details without header."
        enriched = enrich_chunk_content(body, self.PATH)
        assert _strip_header_prefix(enriched, self.PATH) == body


class TestPackageChunk:
    """_package_chunk is the single place that builds a final Chunk from
    raw content + metadata — both chunk_document() and
    SplitOverflowStrategy.apply() go through it instead of each calling
    enrich_chunk_content() themselves."""

    def test_embeds_header_path_from_metadata(self):
        metadata = ChunkMetadata(
            source_file="doc.md",
            page_number=None,
            chunk_index=0,
            header_path="# Main > ## Section",
        )
        chunk = _package_chunk("some text", metadata)

        assert chunk.content == "# Main > ## Section\n\nsome text"
        assert chunk.metadata is metadata

    def test_leaves_content_unchanged_when_no_header_path(self):
        metadata = ChunkMetadata(source_file="doc.pdf", page_number=1, chunk_index=0)
        chunk = _package_chunk("some text", metadata)

        assert chunk.content == "some text"
        assert chunk.metadata is metadata


def test_chunk_document_with_word_header_map(monkeypatch, settings_override):
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNKING_STRATEGY="word", CHUNK_SIZE=3, CHUNK_OVERLAP=0),
    )
    full_text = "## Overview intro text more details here"
    word_page_map = [1] * 7
    word_header_map = ["# Main > ## Overview"] * 7

    chunks = chunk_document(
        full_text,
        word_page_map,
        source_file="guide.md",
        driver=MagicMock(),
        word_header_map=word_header_map,
    )

    assert len(chunks) == 3
    # Chunk 0 has words ['##', 'Overview', 'intro'] -> leaf replaced
    assert chunks[0].content == "# Main > ## Overview\n\nintro"
    assert chunks[0].metadata.header_path == "# Main > ## Overview"

    # Chunk 1 has words ['text', 'more', 'details'] -> header prepended
    assert chunks[1].content == "# Main > ## Overview\n\ntext more details"
    assert chunks[1].metadata.header_path == "# Main > ## Overview"
